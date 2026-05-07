import logging
import os
import pathlib
import tempfile
import time
import zipfile
import csv

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


def _checkpoint_sidecar_path(checkpoint_path: str) -> str:
    root, ext = os.path.splitext(checkpoint_path)
    if not ext:
        ext = ".pt"
    return f"{root}.trainer{ext}"


def _extract_inner_policy_checkpoint(trainer_path: str) -> str:
    path = pathlib.Path(trainer_path)
    sibling_policy_path = path.with_name(path.name.replace(".trainer", "", 1))
    if sibling_policy_path.exists():
        return str(sibling_policy_path)

    with zipfile.ZipFile(path, "r") as archive:
        candidates = [
            name for name in archive.namelist()
            if "/" not in name and name.endswith(".pt") and ".trainer" not in name
        ]
        if not candidates:
            return trainer_path
        with archive.open(candidates[0]) as src, open(sibling_policy_path, "wb") as dst:
            dst.write(src.read())
    logging.info("Extracted policy checkpoint from %s to %s", trainer_path, sibling_policy_path)
    return str(sibling_policy_path)


def _load_resume_sidecar(checkpoint_path: str):
    sidecar_path = _checkpoint_sidecar_path(checkpoint_path)
    if not os.path.exists(sidecar_path) and ".trainer" in checkpoint_path:
        sidecar_path = checkpoint_path
    if not os.path.exists(sidecar_path):
        logging.info("No trainer sidecar state found at %s", sidecar_path)
        return None

    try:
        return torch.load(sidecar_path, map_location="cpu")
    except RuntimeError:
        # Some W&B artifact downloads contain both the policy checkpoint and the
        # trainer checkpoint in one zip. PyTorch requires a single archive root,
        # so rewrite just the trainer entries to a temporary loadable archive.
        with zipfile.ZipFile(sidecar_path, "r") as archive:
            roots = {
                name.split("/", 1)[0]
                for name in archive.namelist()
                if "/" in name and ".trainer" in name.split("/", 1)[0]
            }
            if not roots:
                raise
            root = sorted(roots)[0]
            fd, temp_path = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            try:
                with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED) as clean_archive:
                    for info in archive.infolist():
                        if info.filename.startswith(root + "/"):
                            clean_archive.writestr(info, archive.read(info.filename))
                return torch.load(temp_path, map_location="cpu")
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

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
            # checkpoint_path in task yaml — it would clear algo.checkpoint_path from CLI.
            cfg.algo = OmegaConf.merge(cfg.algo, task_ppo_cfg)
        else:
            logging.warning("ppo_cfg '%s' not found at %s, using default.", ppo_cfg_name, ppo_cfg_path)

    record_video = bool(cfg.get("record_video", False))
    if record_video:
        cfg.sim.enable_replicator = True
        if cfg.get("headless", True):
            logging.warning("record_video=True: forcing headless=false (required for RGB capture).")
        cfg.headless = False

    simulation_app = init_simulation_app(cfg)

    setproctitle(cfg.task.name)
    print(OmegaConf.to_yaml(cfg))

    resume_sidecar_state = None
    checkpoint_path = cfg.algo.get("checkpoint_path", None)
    if checkpoint_path:
        checkpoint_path = str(checkpoint_path)
        if ".trainer" in checkpoint_path:
            cfg.algo.checkpoint_path = _extract_inner_policy_checkpoint(checkpoint_path)
        resume_sidecar_state = _load_resume_sidecar(str(cfg.algo.checkpoint_path))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    if isinstance(resume_sidecar_state, dict):
        env_resume_state = resume_sidecar_state.get("env", {})
        trainer_resume_state = resume_sidecar_state.get("trainer", {})
        if env_resume_state and hasattr(base_env, "load_resume_state"):
            base_env.load_resume_state(env_resume_state)
            logging.info("Restored env resume state from checkpoint sidecar.")
        elif trainer_resume_state and hasattr(base_env, "set_constrained_speed_controller"):
            base_env.set_constrained_speed_controller(
                phase=int(trainer_resume_state.get("controller_phase", 0)),
                speed_pressure=float(trainer_resume_state.get("controller_speed_pressure", 0.0)),
                exploration_pressure=float(
                    trainer_resume_state.get("controller_exploration_pressure", 0.0)
                ),
                collapse_active=bool(
                    trainer_resume_state.get("controller_collapse_active", False)
                ),
                entropy_target=float(trainer_resume_state.get("target_entropy_coef", 0.0)),
            )
            logging.info("Restored controller resume state from checkpoint sidecar.")

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

    frames_per_batch = env.num_envs * int(cfg.algo.get("train_every", 32))

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

    monitor_path = cfg.get("play_stats_path", None)
    monitor_file = None
    monitor_writer = None
    monitor_fields = [
        "batch",
        "env_frames",
        "rollout_fps",
        "done_count",
        "success",
        "gates_passed",
        "furthest_gate",
        "episode_len",
        "lap_time_steps",
        "collision",
        "crashed_z",
        "crashed_z_gate",
        "crashed_distance",
        "final_dist_to_gate",
        "mean_speed",
        "max_speed",
        "mean_altitude",
        "min_altitude",
        "curriculum_phase",
        "controller_speed_pressure",
        "controller_exploration_pressure",
    ] + [f"gate_{idx}_crosses" for idx in range(getattr(base_env, "num_course_gates", 13))]
    if monitor_path:
        monitor_path = os.path.abspath(str(monitor_path))
        monitor_dir = os.path.dirname(monitor_path)
        if monitor_dir:
            os.makedirs(monitor_dir, exist_ok=True)
        monitor_file = open(monitor_path, "w", newline="")
        monitor_writer = csv.DictWriter(monitor_file, fieldnames=monitor_fields)
        monitor_writer.writeheader()
        logging.info("Writing playback gate monitor to %s", monitor_path)

    def _last_value(data, key, default=0.0):
        value = data.get(("next", "stats", key), None)
        if value is None:
            return default
        flat = value.detach().reshape(-1)
        if flat.numel() == 0:
            return default
        return float(flat[-1].float().item())

    def _value_at(data, key, index, default=0.0):
        value = data.get(("next", "stats", key), None)
        if value is None:
            return default
        flat = value.detach().reshape(-1)
        if flat.numel() == 0:
            return default
        safe_index = max(0, min(int(index), flat.numel() - 1))
        return float(flat[safe_index].float().item())

    pbar = tqdm(collector)
    base_env.eval()
    env.eval()
    try:
        for i, data in enumerate(pbar):
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            episode_stats.add(data.to_tensordict())

            if len(episode_stats) >= base_env.num_envs:
                stats = {
                    "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                    for k, v in episode_stats.pop().items(True, True)
                }
                info.update(stats)

            done = data.get(("next", "done"), None)
            done_count = int(done.reshape(-1).sum().item()) if done is not None else 0
            row = {
                "batch": i,
                "env_frames": int(collector._frames),
                "rollout_fps": float(collector._fps),
                "done_count": done_count,
            }
            for key in monitor_fields[4:]:
                row[key] = _last_value(data, key)

            gate_crosses = [
                int(row.get(f"gate_{idx}_crosses", 0.0))
                for idx in range(getattr(base_env, "num_course_gates", 13))
            ]
            print(
                "PLAY_MONITOR "
                f"batch={row['batch']} frames={row['env_frames']} "
                f"gates_passed={row['gates_passed']:.0f} "
                f"furthest_gate={row['furthest_gate']:.0f} "
                f"success={row['success']:.0f} collision={row['collision']:.0f} "
                f"dist={row['final_dist_to_gate']:.2f} "
                f"ep_len={row['episode_len']:.0f} "
                f"phase={row['curriculum_phase']:.0f} "
                f"speed_pressure={row['controller_speed_pressure']:.2f} "
                f"crosses={gate_crosses}",
                flush=True,
            )
            if done_count > 0 and done is not None:
                done_indices = done.detach().reshape(-1).nonzero(as_tuple=False).flatten()
                for label, done_index in (
                    ("first", done_indices[0]),
                    ("last", done_indices[-1]),
                ):
                    gate_crosses_at_done = [
                        int(_value_at(data, f"gate_{idx}_crosses", done_index))
                        for idx in range(getattr(base_env, "num_course_gates", 13))
                    ]
                    print(
                        "PLAY_DONE "
                        f"{label}=1 batch={i} frame_index={int(done_index.item())} "
                        f"gates_passed={_value_at(data, 'gates_passed', done_index):.0f} "
                        f"furthest_gate={_value_at(data, 'furthest_gate', done_index):.0f} "
                        f"success={_value_at(data, 'success', done_index):.0f} "
                        f"collision={_value_at(data, 'collision', done_index):.0f} "
                        f"crashed_z={_value_at(data, 'crashed_z', done_index):.0f} "
                        f"crashed_z_gate={_value_at(data, 'crashed_z_gate', done_index):.0f} "
                        f"crashed_distance={_value_at(data, 'crashed_distance', done_index):.0f} "
                        f"dist={_value_at(data, 'final_dist_to_gate', done_index):.2f} "
                        f"ep_len={_value_at(data, 'episode_len', done_index):.0f} "
                        f"crosses={gate_crosses_at_done}",
                        flush=True,
                    )
            if monitor_writer is not None:
                monitor_writer.writerow(row)
                monitor_file.flush()

            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

            pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})
    finally:
        if monitor_file is not None:
            monitor_file.close()

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
