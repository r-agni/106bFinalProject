import logging
import os
import pathlib

import hydra
import torch

from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType, step_mdp
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


def _checkpoint_sidecar_path(checkpoint_path: str) -> str:
    root, ext = os.path.splitext(checkpoint_path)
    if not ext:
        ext = ".pt"
    return f"{root}.trainer{ext}"


def _manual_episode_rollout(env, policy, max_steps: int, exploration: ExplorationType):
    """Run a single episode with explicit reset/step control.

    Isaac playback was unstable when chaining episodes through env.rollout with
    auto_reset=True. Driving each episode manually keeps resets isolated and
    produces one clean stats snapshot per episode.
    """

    tensordict = env.reset()
    final_stats = None

    with set_exploration_type(exploration):
        for _ in range(max_steps):
            policy_output = policy(tensordict)
            step_td = env.step(policy_output)
            next_td = step_td.get("next")
            final_stats = next_td["stats"].clone().cpu()

            done = next_td.get("done")
            if torch.any(done):
                break

            tensordict = step_mdp(step_td)

    if final_stats is None:
        raise RuntimeError("Playback episode produced no stats.")

    return final_stats


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.play_safe_reset = bool(cfg.get("play_safe_reset", True))

    # Match train.py: merge cfg/algo/<ppo_cfg> when the task specifies it (e.g. DroneRace).
    ppo_cfg_name = cfg.task.get("ppo_cfg", None)
    if ppo_cfg_name:
        algo_dir = pathlib.Path(__file__).resolve().parent.parent / "cfg" / "algo"
        ppo_cfg_path = algo_dir / ppo_cfg_name
        if not ppo_cfg_path.suffix:
            ppo_cfg_path = ppo_cfg_path.with_suffix(".yaml")
        if ppo_cfg_path.exists():
            logging.info("Loading task-specific PPO config: %s", ppo_cfg_path)
            task_ppo_cfg = OmegaConf.load(ppo_cfg_path)
            # Task yaml overrides base algo (e.g. hidden_units). Do not put empty
            # checkpoint_path in task yaml - it would clear algo.checkpoint_path from CLI.
            cfg.algo = OmegaConf.merge(cfg.algo, task_ppo_cfg)
        else:
            logging.warning(
                "ppo_cfg '%s' not found at %s, using default.",
                ppo_cfg_name,
                ppo_cfg_path,
            )

    record_video = bool(cfg.get("record_video", False))
    if record_video:
        cfg.sim.enable_replicator = True
        if cfg.get("headless", True):
            logging.warning(
                "record_video=True: forcing headless=false (required for RGB capture)."
            )
        cfg.headless = False

    simulation_app = init_simulation_app(cfg)

    setproctitle(cfg.task.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    restore_sidecar = bool(cfg.get("play_restore_sidecar", False))
    if (
        restore_sidecar
        and cfg.algo.get("checkpoint_path", None)
        and hasattr(base_env, "load_resume_state")
    ):
        sidecar_path = _checkpoint_sidecar_path(str(cfg.algo.checkpoint_path))
        if os.path.exists(sidecar_path):
            try:
                sidecar_state = torch.load(sidecar_path, map_location="cpu")
                env_state = {}
                if isinstance(sidecar_state, dict):
                    if "env" in sidecar_state and isinstance(sidecar_state["env"], dict):
                        env_state = sidecar_state["env"]
                    elif any(
                        key in sidecar_state
                        for key in (
                            "curriculum_phase",
                            "controller_speed_pressure",
                            "controller_exploration_pressure",
                            "lap_completion_rate_ema",
                        )
                    ):
                        env_state = sidecar_state
                if env_state:
                    base_env.load_resume_state(env_state)
                    logging.info("Loaded environment resume state from %s", sidecar_path)
                else:
                    logging.warning(
                        "Sidecar %s did not contain an environment resume state.",
                        sidecar_path,
                    )
            except Exception as exc:
                logging.warning("Failed to load sidecar %s: %s", sidecar_path, exc)

    # Playback should default to the same full-lap gate-0 reset protocol used by
    # training evaluation metrics, not the curriculum-focused reset mix restored
    # from a sidecar checkpoint.
    if bool(cfg.get("play_disable_reset_curriculum", True)):
        if hasattr(base_env, "reset_curriculum_enabled"):
            base_env.reset_curriculum_enabled = False
        if hasattr(base_env, "active_reset_curriculum_gate"):
            base_env.active_reset_curriculum_gate = -1
        logging.info(
            "Disabled reset curriculum for playback; episodes will start from gate 0."
        )

    transforms = [InitTracker()]

    # A CompositeSpec is by default processed by an entity-based encoder.
    # Ravel it to use an MLP encoder instead.
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(
            base_env.observation_spec, ("agents", "observation_central")
        )
        transforms.append(transform)

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
            transform = AttitudeController(
                base_env.controller, action_key=("agents", "action")
            )
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    # Render every `render_interval` seconds of simulation time (default 0.05 s).
    # The policy still runs every dt * substeps seconds - only the visual update
    # is throttled, so behavior is identical to training.
    render_interval_sec = cfg.get("render_interval", 0.05)
    policy_dt = base_env.cfg.sim.dt * base_env.substeps
    render_every_n_steps = max(1, round(render_interval_sec / policy_dt))
    print(f"render_every_n_steps: {render_every_n_steps}")
    print(f"render_interval_sec: {render_interval_sec}")
    print(f"policy_dt: {policy_dt}")

    step_counter = [0]
    substeps = base_env.substeps

    def render_fn(substep: int) -> bool:
        if substep == substeps - 1:
            step_counter[0] += 1
        return substep == substeps - 1 and (step_counter[0] % render_every_n_steps == 0)

    base_env.enable_render(render_fn)

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device,
        )
    except KeyError as exc:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}") from exc

    frames_per_batch = int(
        cfg.get("play_frames_per_batch", env.num_envs * int(cfg.algo.get("train_every", 32)))
    )
    max_episodes = cfg.get("play_max_episodes", None)
    max_episodes = int(max_episodes) if max_episodes is not None else None

    stats_keys = [
        k
        for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    # Default SyncDataCollector exploration is RANDOM (stochastic actions). For playback,
    # MODE uses the policy distribution's mode (Gaussian mean), which looks cleaner.
    exploration_name = str(cfg.get("play_exploration", "MODE")).upper()
    exploration = getattr(ExplorationType, exploration_name, ExplorationType.MODE)
    use_rollout_playback = bool(
        cfg.get(
            "play_use_rollout",
            max_episodes is not None and base_env.num_envs == 1 and not record_video,
        )
    )
    if use_rollout_playback:
        base_env.eval()
        env.eval()
        completed_episodes = 0
        successful_episodes = 0
        target_episodes = max_episodes if max_episodes is not None else 1

        for episode_idx in range(target_episodes):
            traj_stats = _manual_episode_rollout(
                env=env,
                policy=policy,
                max_steps=base_env.max_episode_length,
                exploration=exploration,
            )
            completed_episodes += base_env.num_envs
            if "success" in traj_stats:
                successful_episodes += int(traj_stats["success"].sum().item())

            info = {
                "episode_index": float(episode_idx + 1),
                "completed_episodes": float(completed_episodes),
                "successful_episodes": float(successful_episodes),
            }
            info.update(
                {
                    "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                    for k, v in traj_stats.items()
                }
            )
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        logging.info(
            "Completed rollout playback for %s episodes (%s successes).",
            completed_episodes,
            successful_episodes,
        )
        simulation_app.close()
        return

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
    completed_episodes = 0
    successful_episodes = 0
    for _, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            popped = episode_stats.pop()
            if popped.batch_size:
                completed_episodes += int(popped.batch_size[0])
                if "success" in popped.keys():
                    successful_episodes += int(popped["success"].sum().item())
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in popped.items(True, True)
            }
            if max_episodes is not None:
                info["completed_episodes"] = float(completed_episodes)
                info["successful_episodes"] = float(successful_episodes)
            info.update(stats)

        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})
        if max_episodes is not None and completed_episodes >= max_episodes:
            logging.info(
                "Stopping after %s completed episodes (%s successes).",
                completed_episodes,
                successful_episodes,
            )
            break

    if hasattr(collector, "shutdown"):
        collector.shutdown()

    if record_video:
        video_path = cfg.get("video_path", "hover_playback.mp4")
        if not os.path.isabs(video_path):
            video_path = os.path.join(os.getcwd(), video_path)
        video_dir = os.path.dirname(video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

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
            "Recording video: max_steps=%s, frame_interval=%s, fps=%s -> %s",
            max_steps,
            frame_interval,
            fps,
            video_path,
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
                "No video frames captured. Ensure viewport + Replicator "
                "(enable_replicator) work on your machine."
            )
        else:
            try:
                import imageio.v2 as imageio

                imageio.mimsave(video_path, frames, fps=fps, codec="libx264")
            except Exception as exc:
                logging.warning("MP4 save failed (%s); trying GIF fallback.", exc)
                gif_path = os.path.splitext(video_path)[0] + ".gif"
                import imageio.v2 as imageio

                imageio.mimsave(gif_path, frames, fps=min(fps, 20))
                video_path = gif_path
            logging.info("Saved recording to %s (%d frames)", video_path, len(frames))

    simulation_app.close()


if __name__ == "__main__":
    main()
