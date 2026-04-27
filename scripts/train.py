import logging
import os
import random
import re
import signal
import time
import yaml
import traceback
from collections import deque

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


def set_global_reproducibility(seed: int, deterministic: bool = True):
    """Seed common RNGs and optionally opt into deterministic torch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)

@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # If the task yaml specifies a ppo_cfg field, load that file from cfg/algo/
    # and merge it over the default algo config, allowing per-task PPO settings.
    ppo_cfg_name = cfg.task.get("ppo_cfg", None)
    if ppo_cfg_name:
        import pathlib
        algo_dir = pathlib.Path(__file__).parent.parent / "cfg" / "algo"
        ppo_cfg_path = algo_dir / ppo_cfg_name
        if not ppo_cfg_path.suffix:
            ppo_cfg_path = ppo_cfg_path.with_suffix(".yaml")
        if ppo_cfg_path.exists():
            logging.info(f"Loading task-specific PPO config: {ppo_cfg_path}")
            task_ppo_cfg = OmegaConf.load(ppo_cfg_path)
            cfg.algo = OmegaConf.merge(cfg.algo, task_ppo_cfg)
        else:
            logging.warning(f"ppo_cfg '{ppo_cfg_name}' not found at {ppo_cfg_path}, using default.")

    set_global_reproducibility(cfg.seed, deterministic=cfg.get("deterministic", False))

    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))
    
    # Save config.yaml early to ensure it's available even if interrupted
    # Wandb saves config.yaml when finish() is called, but we want it saved immediately
    def save_config_early():
        try:
            config_path = os.path.join(run.dir, "files", "config.yaml")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            # Save config in wandb's format: each key has a 'value' field
            config_dict = {}
            for key, value in run.config.items():
                config_dict[key] = {"value": value}
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logging.info(f"Saved config.yaml to {config_path}")
        except Exception as e:
            logging.warning(f"Could not save config.yaml early: {e}")
    
    save_config_early()

    run.summary["curriculum/phase2_trigger_rule"] = (
        "phase==1 and gate0-start lap_completion_rate_ema >= phase2_speed_focus_accuracy_rate"
    )
    run.summary["curriculum/phase2_trigger_threshold"] = float(
        cfg.task.get("phase2_speed_focus_accuracy_rate", 0.40)
    )
    
    # Set up signal handler to ensure config is saved on Ctrl+C
    # The finally block will handle wandb.finish() and simulation_app.close()
    def signal_handler(sig, frame):
        try:
            sig_name = signal.Signals(sig).name
        except Exception:
            sig_name = str(sig)
        logging.warning(f"Received signal {sig_name} ({sig}). Saving config and exiting...")
        save_config_early()  # Save config immediately before cleanup
        # Let the exception propagate to trigger finally block
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

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

    adaptive_entropy_cfg = cfg.algo.get("adaptive_entropy", {})
    adaptive_entropy_enabled = bool(adaptive_entropy_cfg.get("enabled", False))
    adaptive_entropy_enabled = adaptive_entropy_enabled and hasattr(policy, "entropy_coef")
    if bool(cfg.algo.get("adaptive_entropy", {}).get("enabled", False)) and not hasattr(policy, "entropy_coef"):
        logging.warning(
            "adaptive_entropy.enabled is true, but policy %s does not expose entropy_coef; "
            "falling back to fixed entropy.",
            type(policy).__name__,
        )

    if adaptive_entropy_enabled:
        current_entropy_coef = float(adaptive_entropy_cfg.get("phase0_coef_max", cfg.algo.entropy_coef))
        target_entropy_coef = current_entropy_coef
        prev_curriculum_phase = 0.0
        speed_perf_ema = 0.0
        phase1_rebump_applied = False
        policy.entropy_coef = current_entropy_coef
    else:
        current_entropy_coef = float(getattr(policy, "entropy_coef", cfg.algo.entropy_coef))
        target_entropy_coef = current_entropy_coef
        prev_curriculum_phase = 0.0
        speed_perf_ema = 0.0
        phase1_rebump_applied = False
        if hasattr(policy, "entropy_coef"):
            policy.entropy_coef = current_entropy_coef

    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _slew_toward(current: float, target: float, max_delta: float) -> float:
        delta = target - current
        if delta > max_delta:
            return current + max_delta
        if delta < -max_delta:
            return current - max_delta
        return target

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)
    auto_stop_cfg = cfg.get("auto_stop", {})
    auto_stop_enabled = bool(auto_stop_cfg.get("enabled", False))
    auto_stop_stop_on_first_success = bool(auto_stop_cfg.get("stop_on_first_success", False))
    auto_stop_min_frames = int(auto_stop_cfg.get("min_frames", 0))
    auto_stop_window = max(int(auto_stop_cfg.get("window", 30)), 1)
    auto_stop_thresholds = {
        "lap_completion_rate": auto_stop_cfg.get("lap_completion_rate", None),
        "gates_passed_per_ep": auto_stop_cfg.get("gates_passed_per_ep", None),
        "mean_speed_ms": auto_stop_cfg.get("mean_speed_ms", None),
        "min_gate_cross_rate": auto_stop_cfg.get("min_gate_cross_rate", None),
    }
    auto_stop_history = {
        key: deque(maxlen=auto_stop_window)
        for key, threshold in auto_stop_thresholds.items()
        if threshold is not None
    }
    auto_stop_reason = None
    required_gate_count = max(
        int(getattr(base_env, "num_course_gates", getattr(base_env, "num_gates", 1))),
        1,
    )
    logged_curriculum_phase = int(round(float(getattr(base_env, "curriculum_phase", 0))))
    phase2_started_env_frames = None
    phase2_started_iter = None
    phase2_started_accuracy_ema = None

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )
    resume_frame_offset = int(cfg.get("resume_frame_offset", 0))
    if resume_frame_offset <= 0 and cfg.algo.get("checkpoint_path", None):
        match = re.search(r"checkpoint_(\d+)\.pt$", os.path.basename(str(cfg.algo.checkpoint_path)))
        if match:
            resume_frame_offset = int(match.group(1))
    if resume_frame_offset > 0:
        collector._initial_frames = resume_frame_offset
        logging.info("Resuming collector frame count from %s", resume_frame_offset)

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE
    ):
        prev_reset_curriculum_enabled = getattr(base_env, "reset_curriculum_enabled", None)
        render_callback = RenderCallback(interval=2)
        try:
            if prev_reset_curriculum_enabled is not None:
                base_env.reset_curriculum_enabled = False

            base_env.enable_render(True)
            base_env.eval()
            env.eval()
            env.set_seed(seed)
            env.reset()

            with set_exploration_type(exploration_type):
                trajs = env.rollout(
                    max_steps=base_env.max_episode_length,
                    policy=policy,
                    callback=render_callback,
                    auto_reset=True,
                    break_when_any_done=False,
                    return_contiguous=False,
                )

            done = trajs.get(("next", "done"))
            first_done = torch.argmax(done.long(), dim=1).cpu()

            def take_first_episode(tensor: torch.Tensor):
                indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
                return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

            traj_stats = {
                k: take_first_episode(v)
                for k, v in trajs[("next", "stats")].cpu().items()
            }

            info = {
                "eval/stats." + k: torch.mean(v.float()).item()
                for k, v in traj_stats.items()
            }

            video_array = render_callback.get_video_array(axes="t c h w")
            if video_array is not None:
                info["recording"] = wandb.Video(
                    video_array,
                    fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                    format="mp4"
                )

            return info
        finally:
            if prev_reset_curriculum_enabled is not None:
                base_env.reset_curriculum_enabled = prev_reset_curriculum_enabled
            base_env.enable_render(not cfg.headless)
            env.reset()

    loop_exception = None
    try:
        logging.info(
            f"Starting training loop: frames_per_batch={frames_per_batch}, "
            f"total_frames={total_frames}, num_envs={env.num_envs}"
        )
        if auto_stop_enabled:
            logging.info(
                "Auto-stop enabled: stop_on_first_success=%s min_frames=%s thresholds=%s window=%s",
                auto_stop_stop_on_first_success,
                auto_stop_min_frames,
                {k: v for k, v in auto_stop_thresholds.items() if v is not None},
                auto_stop_window,
            )
        pbar = tqdm(collector, total=total_frames//frames_per_batch)
        env.train()
        logging.info("Entering collector iteration.")
        for i, data in enumerate(pbar):
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            should_stop = False
            if adaptive_entropy_enabled:
                info["entropy/current_coef"] = current_entropy_coef
                info["entropy/target_coef"] = target_entropy_coef
                info["entropy/rebump_applied"] = float(phase1_rebump_applied)
            episode_stats.add(data.to_tensordict())

            if len(episode_stats) >= base_env.num_envs:
                raw_stats = episode_stats.pop()
                stats = {
                    "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                    for k, v in raw_stats.items(True, True)
                }
                info.update(stats)

                # --- derived / renamed metrics for easier W&B dashboard reading ---
                def _mean(key):
                    v = raw_stats.get(("stats", key), None)
                    return torch.mean(v.float()).item() if v is not None else None

                def _stats_tensor(key):
                    v = raw_stats.get(("stats", key), None)
                    return v.reshape(-1) if v is not None else None

                def _masked_mean(key, mask):
                    values = _stats_tensor(key)
                    if values is None or mask is None or not bool(mask.any().item()):
                        return None
                    return values.float()[mask].mean().item()

                episode_start_gate = _stats_tensor("episode_start_gate")
                gate0_start_mask = None
                gate0_episode_count = 0
                if episode_start_gate is not None:
                    gate0_start_mask = episode_start_gate.long() == 0
                    gate0_episode_count = int(gate0_start_mask.sum().item())

                success_values = _stats_tensor("success")
                successful_mask = None
                if success_values is not None:
                    successful_mask = success_values.bool()

                gate0_success_mask = None
                gate0_success_count = 0
                if gate0_start_mask is not None and successful_mask is not None:
                    gate0_success_mask = gate0_start_mask & successful_mask
                    gate0_success_count = int(gate0_success_mask.sum().item())

                # Racing performance
                mean_speed   = _mean("mean_speed")
                max_speed    = _mean("max_speed")
                gates_passed = _mean("gates_passed")
                success_rate = _mean("success")
                ep_len       = _mean("episode_len")
                lap_time     = _masked_mean("lap_time_steps", successful_mask)

                task_mean_speed = _masked_mean("mean_speed", gate0_start_mask)
                task_max_speed = _masked_mean("max_speed", gate0_start_mask)
                task_gates_passed = _masked_mean("gates_passed", gate0_start_mask)
                task_success_rate = _masked_mean("success", gate0_start_mask)
                task_ep_len = _masked_mean("episode_len", gate0_start_mask)
                task_completion_time_steps = _masked_mean("lap_time_steps", gate0_success_mask)

                # Crash breakdown
                crash_total    = _mean("collision")
                crash_ground   = _mean("crashed_z")
                crash_contact  = _mean("crashed_z_gate")
                crash_distance = _mean("crashed_distance")

                # Reward components
                rew_progress  = _mean("reward_progress")
                rew_speed     = _mean("reward_speed")
                rew_gates     = _mean("reward_gates")
                rew_stacked_direction = _mean("reward_stacked_direction")
                rew_stacked_bonus = _mean("reward_stacked_bonus")
                rew_penalties = _mean("reward_penalties")
                rew_return    = _mean("return")

                # Behaviour diagnostics
                ang_rate    = _mean("mean_ang_rate")
                decay_frac  = _mean("ang_penalty_decay_frac")
                uprightness = _mean("drone_uprightness")
                truncated   = _mean("truncated")

                # Additional raw stats
                furthest_gate    = _mean("furthest_gate")
                final_dist       = _mean("final_dist_to_gate")
                mean_alt         = _mean("mean_altitude")
                min_alt          = _mean("min_altitude")
                mean_action_mag  = _mean("mean_action_magnitude")
                rew_altitude     = _mean("reward_altitude")
                rew_approach     = _mean("reward_approach")
                rew_centering    = _mean("reward_centering")
                curriculum_phase = _mean("curriculum_phase")
                curriculum_accuracy = _mean("curriculum_accuracy_ema")
                curriculum_furthest = _mean("curriculum_furthest_ema")
                reset_active_gate = _mean("reset_active_gate")
                reset_from_gate0 = _mean("reset_from_gate0")
                reset_from_prev_gate = _mean("reset_from_prev_gate")
                reset_from_random_gate = _mean("reset_from_random_gate")

                derived = {}

                # Racing metrics
                if mean_speed   is not None: derived["race/mean_speed_ms"]       = mean_speed
                if max_speed    is not None: derived["race/max_speed_ms"]         = max_speed
                if gates_passed is not None: derived["race/gates_passed_per_ep"]  = gates_passed
                if success_rate is not None: derived["race/lap_completion_rate"]  = success_rate
                if ep_len       is not None: derived["race/episode_len_steps"]    = ep_len
                if ep_len       is not None: derived["race/episode_len_sec"]      = ep_len * cfg.sim.dt * cfg.sim.substeps
                if lap_time is not None:
                    derived["race/lap_time_steps"] = lap_time
                    derived["race/lap_time_sec"]   = lap_time * cfg.sim.dt * cfg.sim.substeps
                if gate0_episode_count > 0:
                    if task_mean_speed is not None: derived["simple/mean_speed_ms"] = task_mean_speed
                    if task_max_speed is not None: derived["simple/max_speed_ms"] = task_max_speed
                    if task_gates_passed is not None: derived["simple/gates_passed"] = task_gates_passed
                    if task_success_rate is not None:
                        derived["simple/full_course_completion_rate"] = task_success_rate
                        derived["simple/full_course_completion_pct"] = 100.0 * task_success_rate
                    if task_ep_len is not None:
                        derived["simple/episode_time_sec"] = task_ep_len * cfg.sim.dt * cfg.sim.substeps
                    if gate0_success_count > 0 and task_completion_time_steps is not None:
                        derived["simple/completion_time_steps_mean"] = task_completion_time_steps
                        derived["simple/completion_time_sec_mean"] = (
                            task_completion_time_steps * cfg.sim.dt * cfg.sim.substeps
                        )
                if furthest_gate is not None: derived["race/furthest_gate_reached"] = furthest_gate
                if final_dist    is not None: derived["race/final_dist_to_gate_m"]  = final_dist
                # Gates-per-second: how fast the drone is clearing gates
                if gates_passed is not None and ep_len is not None and ep_len > 0:
                    step_sec = cfg.sim.dt * cfg.sim.substeps
                    derived["race/gates_per_second"] = gates_passed / max(ep_len * step_sec, 1e-6)
                gate_cross_rate = None
                if gates_passed is not None:
                    gate_cross_rate = gates_passed / required_gate_count
                    derived["race/gate_cross_rate"] = gate_cross_rate
                task_gate_cross_rate = None
                if task_gates_passed is not None:
                    task_gate_cross_rate = task_gates_passed / required_gate_count

                # Crash breakdown (fraction of episodes)
                if crash_total    is not None: derived["crash/total_rate"]     = crash_total
                if crash_ground   is not None: derived["crash/ground_rate"]    = crash_ground
                if crash_contact  is not None: derived["crash/contact_rate"]   = crash_contact
                if crash_distance is not None: derived["crash/distance_rate"]  = crash_distance

                # Reward components (useful for spotting imbalances)
                if rew_progress  is not None: derived["reward/progress_cumul"]  = rew_progress
                if rew_speed     is not None: derived["reward/speed_cumul"]     = rew_speed
                if rew_gates     is not None: derived["reward/gates_cumul"]     = rew_gates
                if rew_stacked_direction is not None: derived["reward/stacked_direction_cumul"] = rew_stacked_direction
                if rew_stacked_bonus is not None: derived["reward/stacked_bonus_cumul"] = rew_stacked_bonus
                if rew_penalties is not None: derived["reward/penalties_cumul"] = rew_penalties
                if rew_altitude  is not None: derived["reward/altitude_cumul"]  = rew_altitude
                if rew_approach  is not None: derived["reward/approach_cumul"]  = rew_approach
                if rew_centering is not None: derived["reward/centering_cumul"] = rew_centering
                if rew_return    is not None: derived["reward/total_return"]    = rew_return
                if rew_progress is not None and rew_return is not None and abs(rew_return) > 1e-6:
                    derived["reward/progress_fraction"] = rew_progress / rew_return
                # Reward component fractions (diagnostic: which terms dominate)
                if rew_return is not None and abs(rew_return) > 1e-6:
                    stacked_total = (
                        (rew_stacked_direction if rew_stacked_direction is not None else 0.0)
                        + (rew_stacked_bonus if rew_stacked_bonus is not None else 0.0)
                    )
                    if rew_speed     is not None: derived["reward/speed_fraction"]     = rew_speed / rew_return
                    if rew_gates     is not None: derived["reward/gates_fraction"]     = rew_gates / rew_return
                    if rew_stacked_direction is not None or rew_stacked_bonus is not None:
                        derived["reward/stacked_fraction"] = stacked_total / rew_return
                    if rew_altitude  is not None: derived["reward/altitude_fraction"]  = rew_altitude / rew_return
                    if rew_approach  is not None: derived["reward/approach_fraction"]  = rew_approach / rew_return
                    if rew_centering is not None: derived["reward/centering_fraction"] = rew_centering / rew_return
                    if rew_penalties is not None: derived["reward/penalty_fraction"]   = rew_penalties / rew_return

                # Behaviour diagnostics
                if ang_rate         is not None: derived["behaviour/mean_ang_rate_rads"]     = ang_rate
                if decay_frac       is not None: derived["behaviour/ang_penalty_decay_frac"] = decay_frac
                if uprightness      is not None: derived["behaviour/drone_uprightness"]      = uprightness
                if truncated        is not None: derived["behaviour/truncated_rate"]         = truncated
                if mean_action_mag  is not None: derived["behaviour/mean_action_magnitude"]  = mean_action_mag

                # Altitude behaviour
                if mean_alt is not None: derived["behaviour/mean_altitude_m"]  = mean_alt
                if min_alt  is not None: derived["behaviour/min_altitude_m"]   = min_alt

                # Curriculum tracking
                if curriculum_phase is not None: derived["curriculum/phase"] = curriculum_phase
                if curriculum_accuracy is not None: derived["curriculum/accuracy_ema"] = curriculum_accuracy
                if curriculum_furthest is not None: derived["curriculum/furthest_gate_ema"] = curriculum_furthest
                derived["curriculum/speed_focus_accuracy_rate"] = cfg.task.get("phase2_speed_focus_accuracy_rate", 0.40)
                phase2_threshold = float(cfg.task.get("phase2_speed_focus_accuracy_rate", 0.40))
                if curriculum_phase is not None:
                    current_phase = int(round(float(curriculum_phase)))
                    phase2_active = current_phase >= 2
                    phase2_just_unlocked = logged_curriculum_phase < 2 <= current_phase
                    derived["curriculum/phase2_active"] = float(phase2_active)
                    derived["curriculum/phase2_just_unlocked"] = float(phase2_just_unlocked)
                    derived["curriculum/phase2_trigger_threshold"] = phase2_threshold
                    if curriculum_accuracy is not None:
                        derived["curriculum/phase2_trigger_met"] = float(
                            curriculum_accuracy >= phase2_threshold
                        )
                        derived["curriculum/phase2_accuracy_margin"] = (
                            curriculum_accuracy - phase2_threshold
                        )
                    if phase2_just_unlocked:
                        phase2_started_env_frames = int(collector._frames)
                        phase2_started_iter = int(i)
                        phase2_started_accuracy_ema = (
                            float(curriculum_accuracy)
                            if curriculum_accuracy is not None
                            else None
                        )
                        run.summary["curriculum/phase2_started_env_frames"] = phase2_started_env_frames
                        run.summary["curriculum/phase2_started_iter"] = phase2_started_iter
                        if phase2_started_accuracy_ema is not None:
                            run.summary["curriculum/phase2_started_accuracy_ema"] = (
                                phase2_started_accuracy_ema
                            )
                        logging.info(
                            "Curriculum entered phase 2 at env_frames=%s iteration=%s "
                            "accuracy_ema=%s threshold=%s",
                            phase2_started_env_frames,
                            phase2_started_iter,
                            phase2_started_accuracy_ema,
                            phase2_threshold,
                        )
                    if phase2_started_env_frames is not None:
                        derived["curriculum/phase2_started_env_frames"] = float(
                            phase2_started_env_frames
                        )
                        derived["curriculum/phase2_started_iter"] = float(phase2_started_iter)
                        if phase2_started_accuracy_ema is not None:
                            derived["curriculum/phase2_started_accuracy_ema"] = (
                                phase2_started_accuracy_ema
                            )
                    logged_curriculum_phase = current_phase
                speed_phase_scale = 0.0
                if curriculum_phase is not None:
                    if curriculum_phase >= 2.0:
                        speed_phase_scale = cfg.task.get("reward_speed_scale_phase2", cfg.task.get("reward_speed_scale", 0.0))
                    elif curriculum_phase >= 1.0:
                        speed_phase_scale = cfg.task.get("reward_speed_scale", 0.0)
                derived["reward/speed_phase_scale"] = speed_phase_scale
                if reset_active_gate is not None: derived["curriculum/reset_active_gate"] = reset_active_gate
                if reset_from_gate0 is not None: derived["curriculum/reset_from_gate0_rate"] = reset_from_gate0
                if reset_from_prev_gate is not None: derived["curriculum/reset_from_prev_gate_rate"] = reset_from_prev_gate
                if reset_from_random_gate is not None: derived["curriculum/reset_from_random_rate"] = reset_from_random_gate
                for gi in range(12):
                    reset_gate_ema = _mean(f"reset_gate_{gi}_pass_ema")
                    if reset_gate_ema is not None:
                        derived[f"curriculum/reset_gate_{gi:02d}_pass_ema"] = reset_gate_ema
                derived["curriculum/min_accuracy_phase_frames"] = cfg.task.get("curriculum_min_phase_frames", 2_000_000)
                derived["curriculum/speed_unlock_accuracy_rate"] = cfg.task.get("phase_speed_unlock_accuracy_rate", 0.40)
                derived["curriculum/ang_decay_end_frames"] = cfg.task.get("angular_penalty_decay_frames", 100_000_000)

                if adaptive_entropy_enabled:
                    step_sec = cfg.sim.dt * cfg.sim.substeps
                    gates_per_second_batch = 0.0
                    if gates_passed is not None and ep_len is not None and ep_len > 0:
                        gates_per_second_batch = gates_passed / max(ep_len * step_sec, 1e-6)

                    phase0_coef_max = float(adaptive_entropy_cfg.get("phase0_coef_max", 0.005))
                    phase0_coef_min = float(adaptive_entropy_cfg.get("phase0_coef_min", 0.0005))
                    phase1_rebump_coef = float(adaptive_entropy_cfg.get("phase1_rebump_coef", 0.0045))
                    phase1_coef_min = float(adaptive_entropy_cfg.get("phase1_coef_min", 0.0003))
                    speed_ema_alpha = float(adaptive_entropy_cfg.get("speed_ema_alpha", 0.05))
                    speed_signal_low = float(adaptive_entropy_cfg.get("speed_signal_low", 0.05))
                    speed_signal_high = float(adaptive_entropy_cfg.get("speed_signal_high", 0.60))
                    max_delta_per_update = float(adaptive_entropy_cfg.get("max_delta_per_update", 0.00025))

                    curr_phase = float(curriculum_phase if curriculum_phase is not None else prev_curriculum_phase)
                    accuracy_signal = float(curriculum_accuracy if curriculum_accuracy is not None else 0.0)

                    if prev_curriculum_phase == 0.0 and curr_phase >= 1.0:
                        current_entropy_coef = phase1_rebump_coef
                        target_entropy_coef = phase1_rebump_coef
                        phase1_rebump_applied = True
                    elif curr_phase < 1.0:
                        unlock_rate = float(cfg.task.get("phase_speed_unlock_accuracy_rate", 0.40))
                        accuracy_norm = _clamp01(accuracy_signal / max(unlock_rate, 1e-6))
                        target_entropy_coef = phase0_coef_max - accuracy_norm * (phase0_coef_max - phase0_coef_min)
                        current_entropy_coef = _slew_toward(
                            current_entropy_coef, target_entropy_coef, max_delta_per_update
                        )
                    else:
                        speed_perf_ema = (1.0 - speed_ema_alpha) * speed_perf_ema + speed_ema_alpha * gates_per_second_batch
                        speed_span = max(speed_signal_high - speed_signal_low, 1e-6)
                        speed_norm = _clamp01((speed_perf_ema - speed_signal_low) / speed_span)
                        target_entropy_coef = phase1_rebump_coef - speed_norm * (phase1_rebump_coef - phase1_coef_min)
                        current_entropy_coef = _slew_toward(
                            current_entropy_coef, target_entropy_coef, max_delta_per_update
                        )

                    prev_curriculum_phase = curr_phase
                    policy.entropy_coef = current_entropy_coef

                    derived["entropy/current_coef"] = current_entropy_coef
                    derived["entropy/target_coef"] = target_entropy_coef
                    derived["entropy/phase"] = curr_phase
                    derived["entropy/accuracy_signal"] = accuracy_signal
                    derived["entropy/gates_per_second_ema"] = speed_perf_ema
                    derived["entropy/rebump_applied"] = float(phase1_rebump_applied)

                # Per-gate crossing heatmap data — log as individual metrics for WandB bar chart
                for gi in range(12):
                    v = _mean(f"gate_{gi}_crosses")
                    if v is not None:
                        derived[f"gates/gate_{gi:02d}_crosses"] = v

                # WandB bar chart for per-gate distribution
                gate_counts = [_mean(f"gate_{gi}_crosses") or 0.0 for gi in range(12)]
                gate_labels = [f"G{gi}" for gi in range(12)]
                gate_table = wandb.Table(columns=["gate", "mean_crosses"], data=[[label, val] for label, val in zip(gate_labels, gate_counts)])
                derived["gates/crossing_distribution"] = wandb.plot.bar(gate_table, "gate", "mean_crosses", title="Gate Crossing Distribution")

                info.update(derived)

                if auto_stop_enabled:
                    frames_ready = collector._frames >= auto_stop_min_frames
                    if (
                        auto_stop_stop_on_first_success
                        and task_success_rate is not None
                        and task_success_rate > 0.0
                        and frames_ready
                    ):
                        should_stop = True
                        auto_stop_reason = (
                            f"observed gate-0 lap completion rate {task_success_rate:.4f} "
                            f"at env_frames={collector._frames}"
                        )
                    else:
                        current_auto_stop_metrics = {
                            "lap_completion_rate": task_success_rate,
                            "gates_passed_per_ep": task_gates_passed,
                            "mean_speed_ms": task_mean_speed,
                            "min_gate_cross_rate": task_gate_cross_rate,
                        }
                        for key, history in auto_stop_history.items():
                            value = current_auto_stop_metrics.get(key)
                            if value is not None:
                                history.append(float(value))
                            if history:
                                info[f"auto_stop/window_{key}"] = sum(history) / len(history)

                        window_ready = auto_stop_history and all(
                            len(history) == auto_stop_window for history in auto_stop_history.values()
                        )
                        if frames_ready and window_ready:
                            threshold_failures = []
                            for key, threshold in auto_stop_thresholds.items():
                                if threshold is None:
                                    continue
                                window_mean = sum(auto_stop_history[key]) / len(auto_stop_history[key])
                                if window_mean < float(threshold):
                                    threshold_failures.append((key, window_mean, float(threshold)))
                            if not threshold_failures:
                                should_stop = True
                                auto_stop_reason = (
                                    f"windowed auto-stop thresholds met at env_frames={collector._frames}"
                                )

                    info["auto_stop/enabled"] = 1.0
                    if should_stop:
                        info["auto_stop/triggered"] = 1.0

            if hasattr(policy, "entropy_coef"):
                policy.entropy_coef = current_entropy_coef
            info.update(policy.train_op(data.to_tensordict()))

            if eval_interval > 0 and i % eval_interval == 0:
                logging.info(f"Eval at {collector._frames} steps.")
                # info.update(evaluate(seed=cfg.seed))
                info.update(evaluate())
                env.train()
                base_env.train()

            if save_interval > 0 and i % save_interval == 0:
                try:
                    ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                    torch.save(policy.state_dict(), ckpt_path)
                    logging.info(f"Saved checkpoint to {str(ckpt_path)}")
                except AttributeError:
                    logging.warning(f"Policy {policy} does not implement `.state_dict()`")

            run.log(info)
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
            print(f"[train] epoch={i + 1} total_frames_processed={collector._frames}")

            pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

            if should_stop:
                logging.info("Auto-stop triggered: %s", auto_stop_reason)
                break

            if max_iters > 0 and i >= max_iters - 1:
                break

        logging.info("Collector iteration ended normally.")
        logging.info(f"Final Eval at {collector._frames} steps.")
        info = {"env_frames": collector._frames}
        try:
            info.update(evaluate(seed=cfg.seed))
            run.log(info)
        except Exception as eval_error:
            logging.warning(f"Final evaluation/video logging failed: {eval_error}")

        try:
            ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
            torch.save(policy.state_dict(), ckpt_path)

            model_artifact = wandb.Artifact(
                f"{cfg.task.name}-{cfg.algo.name.lower()}",
                type="model",
                description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
                metadata=dict(cfg))

            model_artifact.add_file(ckpt_path)
            wandb.save(ckpt_path)
            run.log_artifact(model_artifact)

            logging.info(f"Saved checkpoint to {str(ckpt_path)}")
        except AttributeError:
            logging.warning(f"Policy {policy} does not implement `.state_dict()`")
    except BaseException as e:
        loop_exception = e
        logging.error(
            f"Training aborted with {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        raise
    finally:
        if loop_exception is None:
            logging.warning("Training loop reached cleanup without a caught exception.")
        else:
            logging.warning(
                f"Cleaning up after exception: {type(loop_exception).__name__}"
            )
        # Ensure config is saved and wandb is finished even on interruption
        save_config_early()
        wandb.finish()
        simulation_app.close()


if __name__ == "__main__":
    main()
