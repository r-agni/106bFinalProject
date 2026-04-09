import logging
import os
import time

import hydra
import torch

from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector, RenderCallback
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.torchrl import EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


FILE_PATH = os.path.dirname(__file__)

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    record_video = bool(cfg.get("record_video", False))
    if record_video:
        cfg.sim.enable_replicator = True
        if cfg.get("headless", True):
            logging.warning("record_video=True: forcing headless=false (required for RGB capture).")
        cfg.headless = False

    simulation_app = init_simulation_app(cfg)

    setproctitle(cfg.task.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # if cfg.task.get("history", False):
    #     # transforms.append(History([("info", "drone_state"), ("info", "prev_action")]))
    #     transforms.append(History([("agents", "observation")]))

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform == "rate":
            if not hasattr(base_env, "controller") or base_env.controller is None:
                raise ValueError(
                    "task.action_transform=rate requires cfg.task.drone_model.controller "
                    "to be set (e.g. RateController)."
                )
            transform = RateController(base_env.controller, action_key=("agents", "action"))
            transforms.append(transform)
        elif action_transform == "attitude":
            if not hasattr(base_env, "controller") or base_env.controller is None:
                raise ValueError(
                    "task.action_transform=attitude requires cfg.task.drone_model.controller "
                    "to be set (e.g. AttitudeController)."
                )
            transform = AttitudeController(base_env.controller, action_key=("agents", "action"))
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    # Render every `render_interval` seconds of simulation time (default 0.1 s).
    # The policy still runs every dt * substeps seconds — only the visual update
    # is throttled, so behaviour is identical to training.
    render_interval_sec = cfg.get("render_interval", 0.05)
    policy_dt = base_env.cfg.sim.dt * base_env.substeps
    render_every_n_steps = max(1, round(render_interval_sec / policy_dt))
    print(f"render_every_n_steps: {render_every_n_steps}")
    print(f"render_interval_sec: {render_interval_sec}")
    print(f"policy_dt: {policy_dt}")
    
    _step_counter = [0]
    _substeps = base_env.substeps

    def _render_fn(substep: int) -> bool:
        if substep == _substeps - 1:
            _step_counter[0] += 1
        return substep == _substeps - 1 and (_step_counter[0] % render_every_n_steps == 0)

    base_env.enable_render(_render_fn)

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    frames_per_batch = env.num_envs * 32

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    # Default SyncDataCollector exploration is RANDOM (stochastic actions). For playback,
    # MODE uses the policy distribution's mode (Gaussian mean), which looks like a stable hover.
    exploration_name = str(cfg.get("play_exploration", "MODE")).upper()
    exploration = getattr(ExplorationType, exploration_name, ExplorationType.MODE)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=cfg.total_frames,
        device=cfg.sim.device,
        return_same_td=True,
        exploration_type=exploration,
    )

    pbar = tqdm(collector)
    base_env.eval()
    env.eval()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

    if record_video:
        video_path = cfg.get("video_path", "hover_playback.mp4")
        if not os.path.isabs(video_path):
            video_path = os.path.join(os.getcwd(), video_path)
        vdir = os.path.dirname(video_path)
        if vdir:
            os.makedirs(vdir, exist_ok=True)

        max_steps = int(cfg.get("record_video_max_steps", 800))
        if cfg.get("total_frames", 0) and cfg.total_frames > 0:
            max_steps = min(max_steps, int(cfg.total_frames))

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(cfg.seed)
        frame_interval = max(1, int(cfg.get("video_frame_interval", 2)))
        render_cb = RenderCallback(interval=frame_interval)
        fps = float(cfg.get("video_fps", 30))

        logging.info(
            f"Recording video: max_steps={max_steps}, frame_interval={frame_interval}, fps={fps} -> {video_path}"
        )
        with set_exploration_type(ExplorationType.MODE):
            env.rollout(
                max_steps=max_steps,
                policy=policy,
                callback=render_cb,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )

        base_env.enable_render(not cfg.headless)
        base_env.train()
        env.train()

        frames = render_cb.get_video_array(axes="t h w c")
        if frames is None or frames.size == 0:
            logging.error(
                "No video frames captured. Ensure viewport + Replicator (enable_replicator) work on your machine."
            )
        else:
            try:
                import imageio.v2 as imageio

                imageio.mimsave(video_path, frames, fps=fps, codec="libx264")
            except Exception as e:
                logging.warning("MP4 save failed (%s); trying GIF fallback.", e)
                gif_path = os.path.splitext(video_path)[0] + ".gif"
                import imageio.v2 as imageio

                imageio.mimsave(gif_path, frames, fps=min(fps, 20))
                video_path = gif_path
            logging.info("Saved recording to %s (%d frames)", video_path, len(frames))

    simulation_app.close()


if __name__ == "__main__":
    main()
