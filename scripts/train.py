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


def _checkpoint_sidecar_path(checkpoint_path: str) -> str:
    root, ext = os.path.splitext(checkpoint_path)
    if not ext:
        ext = ".pt"
    return f"{root}.trainer{ext}"


def _should_log_wandb_stat_key(key) -> bool:
    flat_key = ".".join(key) if isinstance(key, tuple) else str(key)
    return not (
        flat_key == "stats.cheating"
        or flat_key.startswith("stats.cheating_gate_")
    )


def _recursive_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _recursive_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_recursive_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_recursive_to_cpu(v) for v in value)
    return value

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

    # Start WandB before Isaac Sim boot so resumed runs appear online even while
    # extension sync and simulator startup are still in progress.
    run = init_wandb(cfg)
    setproctitle(run.name)
    simulation_app = init_simulation_app(cfg)
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
        "controller-driven: phase==1, stabilize_window stays true, and "
        "controller_phase2_ready_windows reaches controller_plateau_windows"
    )
    run.summary["curriculum/phase2_trigger_threshold"] = float(
        cfg.task.get("controller_plateau_windows", 8)
    )
    run.summary["curriculum/phase2_accuracy_gate_rule"] = (
        "legacy heuristic: gate0-start lap_completion_rate_ema >= "
        "phase2_speed_focus_accuracy_rate"
    )
    run.summary["curriculum/phase2_accuracy_gate_threshold"] = float(
        cfg.task.get("phase2_speed_focus_accuracy_rate", 0.40)
    )
    run.summary["curriculum/phase3_trigger_rule"] = (
        "controller-driven: phase==2, high speed pressure, stable plateau, "
        "acceptable completion/final-gate/crash metrics, and valid t20"
    )
    run.summary["curriculum/phase3_trigger_threshold"] = float(
        cfg.task.get("controller_phase3_lock_plateau_windows", 12)
    )
    run.summary["gates_human/finish_line_note"] = (
        "G13 is the finish line; full-lap success requires crossing it."
    )
    race_objective_completion_target = 0.90
    race_objective_lap_time_target_sec = 13.0
    run.summary["race_objective/target_completion_rate"] = (
        race_objective_completion_target
    )
    run.summary["race_objective/target_lap_time_sec"] = (
        race_objective_lap_time_target_sec
    )
    run.summary["race_objective/goal"] = (
        ">=90% gate0-start full-lap completion with <=13.0s full-lap time"
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

    resume_sidecar_state = None
    loaded_policy_resume_state = {}
    loaded_trainer_resume_state = {}
    loaded_env_resume_state = {}
    if cfg.algo.get("checkpoint_path", None):
        resume_sidecar_path = _checkpoint_sidecar_path(str(cfg.algo.checkpoint_path))
        run.summary["resume/sidecar_path"] = resume_sidecar_path
        if os.path.exists(resume_sidecar_path):
            try:
                resume_sidecar_state = torch.load(
                    resume_sidecar_path, map_location="cpu"
                )
                logging.info(
                    "Loaded trainer sidecar state from %s", resume_sidecar_path
                )
                run.summary["resume/sidecar_loaded"] = 1.0
            except Exception as e:
                logging.warning(
                    "Could not load trainer sidecar state from %s: %s",
                    resume_sidecar_path,
                    e,
                )
                run.summary["resume/sidecar_loaded"] = 0.0
        else:
            logging.info("No trainer sidecar state found at %s", resume_sidecar_path)
            run.summary["resume/sidecar_loaded"] = 0.0

    if isinstance(resume_sidecar_state, dict):
        if any(
            key in resume_sidecar_state for key in ("policy", "trainer", "env")
        ):
            loaded_policy_resume_state = dict(
                resume_sidecar_state.get("policy", {}) or {}
            )
            loaded_trainer_resume_state = dict(
                resume_sidecar_state.get("trainer", {}) or {}
            )
            loaded_env_resume_state = dict(resume_sidecar_state.get("env", {}) or {})
        else:
            loaded_trainer_resume_state = dict(resume_sidecar_state)

    for attr_name, attr_state in loaded_policy_resume_state.items():
        target = getattr(policy, attr_name, None)
        if target is None or not hasattr(target, "load_state_dict"):
            continue
        try:
            target.load_state_dict(attr_state)
        except Exception as e:
            logging.warning(
                "Could not restore policy sidecar state for %s: %s", attr_name, e
            )

    adaptive_entropy_cfg = cfg.algo.get("adaptive_entropy", {})
    entropy_controller_enabled = bool(adaptive_entropy_cfg.get("enabled", False))
    entropy_controller_enabled = entropy_controller_enabled and hasattr(policy, "entropy_coef")
    if bool(cfg.algo.get("adaptive_entropy", {}).get("enabled", False)) and not hasattr(policy, "entropy_coef"):
        logging.warning(
            "adaptive_entropy.enabled is true, but policy %s does not expose entropy_coef; "
            "falling back to fixed entropy.",
            type(policy).__name__,
        )

    phase0_coef_max = float(adaptive_entropy_cfg.get("phase0_coef_max", cfg.algo.entropy_coef))
    phase1_min_coef = float(
        adaptive_entropy_cfg.get(
            "phase1_min_coef",
            adaptive_entropy_cfg.get("phase0_coef_min", 0.006),
        )
    )
    phase2_min_coef = float(
        adaptive_entropy_cfg.get(
            "phase2_min_coef",
            adaptive_entropy_cfg.get("phase1_coef_min", 0.002),
        )
    )
    phase2_plateau_base_coef = float(
        adaptive_entropy_cfg.get("phase2_plateau_base_coef", 0.0045)
    )
    phase2_plateau_gain = float(
        adaptive_entropy_cfg.get("phase2_plateau_gain", 0.0030)
    )
    phase2_plateau_max_coef = float(
        adaptive_entropy_cfg.get("phase2_plateau_max_coef", 0.0075)
    )
    phase2_plateau_max_increase_step = float(
        adaptive_entropy_cfg.get("phase2_plateau_max_increase_per_update", 0.0005)
    )
    recovery_coef = float(
        adaptive_entropy_cfg.get(
            "recovery_coef",
            adaptive_entropy_cfg.get("phase1_rebump_coef", 0.010),
        )
    )
    phase3_lock_coef = float(
        adaptive_entropy_cfg.get("phase3_lock_coef", 0.0018)
    )
    phase3_rebound_cap = max(
        float(adaptive_entropy_cfg.get("phase3_rebound_cap", recovery_coef)),
        phase3_lock_coef,
    )
    phase3_max_decrease_step = float(
        adaptive_entropy_cfg.get("phase3_max_decrease_per_update", 0.0010)
    )
    phase3_max_increase_step = float(
        adaptive_entropy_cfg.get("phase3_max_increase_per_update", 0.00025)
    )
    entropy_increase_step = float(
        adaptive_entropy_cfg.get(
            "max_increase_per_update",
            adaptive_entropy_cfg.get("max_delta_per_update", 0.0015),
        )
    )
    entropy_decrease_step = float(
        adaptive_entropy_cfg.get(
            "max_decrease_per_update",
            adaptive_entropy_cfg.get("max_delta_per_update", 0.0005),
        )
    )
    current_entropy_coef = float(getattr(policy, "entropy_coef", cfg.algo.entropy_coef))
    if entropy_controller_enabled:
        current_entropy_coef = phase0_coef_max
    target_entropy_coef = current_entropy_coef
    if hasattr(policy, "entropy_coef"):
        policy.entropy_coef = current_entropy_coef

    controller_completion_target = float(getattr(base_env, "controller_completion_target", 0.78))
    controller_completion_floor = float(getattr(base_env, "controller_completion_floor", 0.70))
    controller_completion_recover = float(getattr(base_env, "controller_completion_recover", 0.60))
    controller_hard_gate_target = float(getattr(base_env, "controller_hard_gate_target", 0.82))
    controller_final_gate_target = float(getattr(base_env, "controller_final_gate_target", 0.78))
    controller_crash_target = float(getattr(base_env, "controller_crash_target", 0.40))
    controller_crash_recover = float(getattr(base_env, "controller_crash_recover", 0.60))
    controller_speed_up_step = float(getattr(base_env, "controller_speed_up_step", 0.015))
    controller_speed_down_step = float(getattr(base_env, "controller_speed_down_step", 0.08))
    controller_speed_panic_step = float(getattr(base_env, "controller_speed_panic_step", 0.20))
    controller_plateau_windows = max(int(getattr(base_env, "controller_plateau_windows", 8)), 1)
    controller_plateau_improvement_eps = float(
        getattr(base_env, "controller_plateau_improvement_eps", 0.01)
    )
    controller_success_min_for_t20 = max(
        int(getattr(base_env, "controller_success_min_for_t20", 32)),
        1,
    )
    controller_ema_alpha = float(cfg.task.get("controller_metric_alpha", 0.08))
    controller_metric_window = max(
        int(cfg.task.get("controller_metric_window", controller_plateau_windows)),
        1,
    )
    controller_stabilize_completion_target = float(
        cfg.task.get(
            "controller_stabilize_completion_target",
            max(controller_completion_target, 0.90),
        )
    )
    controller_stabilize_hard_gate_target = float(
        cfg.task.get(
            "controller_stabilize_hard_gate_target",
            max(controller_hard_gate_target, 0.90),
        )
    )
    controller_stabilize_final_gate_target = float(
        cfg.task.get(
            "controller_stabilize_final_gate_target",
            max(controller_final_gate_target, 0.90),
        )
    )
    controller_stabilize_crash_target = float(
        cfg.task.get("controller_stabilize_crash_target", min(controller_crash_target, 0.12))
    )
    controller_phase2_completion_target = float(
        cfg.task.get(
            "controller_phase2_completion_target",
            max(controller_completion_floor, 0.84),
        )
    )
    controller_phase2_hard_gate_target = float(
        cfg.task.get("controller_phase2_hard_gate_target", 0.86)
    )
    controller_phase2_final_gate_target = float(
        cfg.task.get("controller_phase2_final_gate_target", 0.84)
    )
    controller_phase2_completion_floor = float(
        cfg.task.get(
            "controller_phase2_completion_floor",
            max(controller_completion_floor, 0.76),
        )
    )
    controller_phase2_final_gate_floor = float(
        cfg.task.get("controller_phase2_final_gate_floor", 0.78)
    )
    controller_phase2_crash_target = float(
        cfg.task.get("controller_phase2_crash_target", 0.18)
    )
    controller_phase2_crash_recover = float(
        cfg.task.get("controller_phase2_crash_recover", 0.28)
    )
    controller_phase1_initial_speed_pressure = float(
        cfg.task.get("controller_phase1_initial_speed_pressure", 0.10)
    )
    controller_phase2_initial_speed_pressure = float(
        cfg.task.get("controller_phase2_initial_speed_pressure", 0.35)
    )
    controller_phase2_speed_up_step = float(
        cfg.task.get("controller_phase2_speed_up_step", 0.03)
    )
    controller_phase2_speed_down_step = float(
        cfg.task.get("controller_phase2_speed_down_step", 0.06)
    )
    controller_phase2_panic_step = float(
        cfg.task.get("controller_phase2_panic_step", 0.18)
    )
    controller_phase2_plateau_start_pressure = float(
        cfg.task.get("controller_phase2_plateau_start_pressure", 0.45)
    )
    controller_phase3_lock_speed_pressure = float(
        cfg.task.get("controller_phase3_lock_speed_pressure", 0.95)
    )
    controller_phase3_lock_plateau_windows = max(
        int(cfg.task.get("controller_phase3_lock_plateau_windows", 12)),
        1,
    )
    controller_phase3_lock_completion_target = float(
        cfg.task.get("controller_phase3_lock_completion_target", 0.80)
    )
    controller_phase3_lock_final_gate_target = float(
        cfg.task.get("controller_phase3_lock_final_gate_target", 0.78)
    )
    controller_phase3_lock_crash_target = float(
        cfg.task.get("controller_phase3_lock_crash_target", 0.22)
    )
    controller_phase3_fallback_patience_windows = max(
        int(cfg.task.get("controller_phase3_fallback_patience_windows", 1)),
        1,
    )
    controller_phase3_sticky_after_first_lock = bool(
        cfg.task.get("controller_phase3_sticky_after_first_lock", False)
    )
    controller_phase3_completion_floor = float(
        cfg.task.get("controller_phase3_completion_floor", 0.74)
    )
    controller_phase3_final_gate_floor = float(
        cfg.task.get("controller_phase3_final_gate_floor", 0.72)
    )
    controller_phase3_crash_recover = float(
        cfg.task.get("controller_phase3_crash_recover", 0.30)
    )
    controller_phase3_completion_drop_recover = float(
        cfg.task.get("controller_phase3_completion_drop_recover", 0.12)
    )
    controller_completion_drop_recover = float(
        cfg.task.get("controller_completion_drop_recover", 0.10)
    )
    controller_min_phase_frames = int(
        cfg.task.get(
            "controller_min_phase_frames",
            cfg.task.get("curriculum_min_phase_frames", 0),
        )
    )
    controller_phase = 0
    controller_speed_pressure = 0.0
    controller_exploration_pressure = 0.0
    controller_collapse_active = False
    controller_completion_ema = 0.0
    controller_gate0_crash_ema = 0.0
    controller_completion_signal = 0.0
    controller_crash_signal = 0.0
    controller_t20_lap_time_sec = None
    controller_t20_best_stable_lap_time_sec = None
    controller_phase2_ready_count = 0
    controller_plateau_count = 0
    controller_phase3_has_locked = False
    controller_phase3_bad_window_count = 0
    controller_completion_history = deque(maxlen=6)
    controller_completion_recent = deque(maxlen=controller_metric_window)
    controller_gate0_crash_recent = deque(maxlen=controller_metric_window)
    controller_phase_start_frames = 0
    if loaded_trainer_resume_state:
        controller_phase = int(
            loaded_trainer_resume_state.get("controller_phase", controller_phase)
        )
        controller_speed_pressure = float(
            loaded_trainer_resume_state.get(
                "controller_speed_pressure", controller_speed_pressure
            )
        )
        controller_exploration_pressure = float(
            loaded_trainer_resume_state.get(
                "controller_exploration_pressure",
                controller_exploration_pressure,
            )
        )
        controller_collapse_active = bool(
            loaded_trainer_resume_state.get(
                "controller_collapse_active", controller_collapse_active
            )
        )
        controller_completion_ema = float(
            loaded_trainer_resume_state.get(
                "controller_completion_ema", controller_completion_ema
            )
        )
        controller_gate0_crash_ema = float(
            loaded_trainer_resume_state.get(
                "controller_gate0_crash_ema", controller_gate0_crash_ema
            )
        )
        controller_completion_signal = float(
            loaded_trainer_resume_state.get(
                "controller_completion_signal", controller_completion_signal
            )
        )
        controller_crash_signal = float(
            loaded_trainer_resume_state.get(
                "controller_crash_signal", controller_crash_signal
            )
        )
        controller_t20_lap_time_sec = loaded_trainer_resume_state.get(
            "controller_t20_lap_time_sec", controller_t20_lap_time_sec
        )
        controller_t20_best_stable_lap_time_sec = loaded_trainer_resume_state.get(
            "controller_t20_best_stable_lap_time_sec",
            controller_t20_best_stable_lap_time_sec,
        )
        controller_phase2_ready_count = int(
            loaded_trainer_resume_state.get(
                "controller_phase2_ready_count", controller_phase2_ready_count
            )
        )
        controller_plateau_count = int(
            loaded_trainer_resume_state.get(
                "controller_plateau_count", controller_plateau_count
            )
        )
        controller_phase3_has_locked = bool(
            loaded_trainer_resume_state.get(
                "controller_phase3_has_locked", controller_phase >= 3
            )
        )
        controller_phase3_bad_window_count = max(
            int(
                loaded_trainer_resume_state.get(
                    "controller_phase3_bad_window_count",
                    controller_phase3_bad_window_count,
                )
            ),
            0,
        )
        controller_phase_start_frames = int(
            loaded_trainer_resume_state.get(
                "controller_phase_start_frames", controller_phase_start_frames
            )
        )
        controller_completion_history = deque(
            [
                float(v)
                for v in loaded_trainer_resume_state.get(
                    "controller_completion_history", []
                )
            ],
            maxlen=controller_completion_history.maxlen,
        )
        controller_completion_recent = deque(
            [
                float(v)
                for v in loaded_trainer_resume_state.get(
                    "controller_completion_recent", []
                )
            ],
            maxlen=controller_metric_window,
        )
        controller_gate0_crash_recent = deque(
            [
                float(v)
                for v in loaded_trainer_resume_state.get(
                    "controller_gate0_crash_recent", []
                )
            ],
            maxlen=controller_metric_window,
        )
        current_entropy_coef = float(
            loaded_trainer_resume_state.get(
                "current_entropy_coef", current_entropy_coef
            )
        )
        target_entropy_coef = float(
            loaded_trainer_resume_state.get(
                "target_entropy_coef", target_entropy_coef
            )
        )
    base_env.set_constrained_speed_controller(
        phase=controller_phase,
        speed_pressure=controller_speed_pressure,
        exploration_pressure=controller_exploration_pressure,
        collapse_active=controller_collapse_active,
        entropy_target=target_entropy_coef,
    )
    if loaded_env_resume_state and hasattr(base_env, "load_resume_state"):
        base_env.load_resume_state(loaded_env_resume_state)
        base_env.set_constrained_speed_controller(
            phase=controller_phase,
            speed_pressure=controller_speed_pressure,
            exploration_pressure=controller_exploration_pressure,
            collapse_active=controller_collapse_active,
            entropy_target=target_entropy_coef,
        )
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

    def _slew_toward_asymmetric(
        current: float,
        target: float,
        max_increase: float,
        max_decrease: float,
    ) -> float:
        delta = target - current
        if delta > max_increase:
            return current + max_increase
        if delta < -max_decrease:
            return current - max_decrease
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
    phase3_started_env_frames = None
    phase3_started_iter = None
    if loaded_trainer_resume_state:
        logged_curriculum_phase = int(
            loaded_trainer_resume_state.get(
                "logged_curriculum_phase", logged_curriculum_phase
            )
        )
        phase2_started_env_frames = loaded_trainer_resume_state.get(
            "phase2_started_env_frames", phase2_started_env_frames
        )
        if phase2_started_env_frames is not None:
            phase2_started_env_frames = int(phase2_started_env_frames)
        phase2_started_iter = loaded_trainer_resume_state.get(
            "phase2_started_iter", phase2_started_iter
        )
        if phase2_started_iter is not None:
            phase2_started_iter = int(phase2_started_iter)
        phase2_started_accuracy_ema = loaded_trainer_resume_state.get(
            "phase2_started_accuracy_ema", phase2_started_accuracy_ema
        )
        if phase2_started_env_frames is not None:
            run.summary["curriculum/phase2_started_env_frames"] = (
                phase2_started_env_frames
            )
        if phase2_started_iter is not None:
            run.summary["curriculum/phase2_started_iter"] = phase2_started_iter
        if phase2_started_accuracy_ema is not None:
            run.summary["curriculum/phase2_started_accuracy_ema"] = (
                phase2_started_accuracy_ema
            )
        phase3_started_env_frames = loaded_trainer_resume_state.get(
            "phase3_started_env_frames", phase3_started_env_frames
        )
        if phase3_started_env_frames is not None:
            phase3_started_env_frames = int(phase3_started_env_frames)
            run.summary["curriculum/phase3_started_env_frames"] = (
                phase3_started_env_frames
            )
        phase3_started_iter = loaded_trainer_resume_state.get(
            "phase3_started_iter", phase3_started_iter
        )
        if phase3_started_iter is not None:
            phase3_started_iter = int(phase3_started_iter)
            run.summary["curriculum/phase3_started_iter"] = phase3_started_iter
    if (
        controller_phase >= 3
        or phase3_started_env_frames is not None
        or logged_curriculum_phase >= 3
    ):
        controller_phase3_has_locked = True
    if controller_phase < 3:
        controller_phase3_bad_window_count = 0
    if controller_phase3_has_locked:
        current_entropy_coef = min(current_entropy_coef, phase3_rebound_cap)
        target_entropy_coef = min(target_entropy_coef, phase3_rebound_cap)
    if (
        controller_phase3_sticky_after_first_lock
        and controller_phase3_has_locked
        and controller_phase < 2
    ):
        controller_phase = 2
        controller_exploration_pressure = 0.0
    base_env.set_constrained_speed_controller(
        phase=controller_phase,
        speed_pressure=controller_speed_pressure,
        exploration_pressure=controller_exploration_pressure,
        collapse_active=controller_collapse_active,
        entropy_target=target_entropy_coef,
    )
    if hasattr(policy, "entropy_coef"):
        policy.entropy_coef = current_entropy_coef

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
    if resume_frame_offset <= 0 and loaded_trainer_resume_state:
        resume_frame_offset = int(
            loaded_trainer_resume_state.get("collector_frames", resume_frame_offset)
        )
    if resume_frame_offset > 0:
        collector._initial_frames = resume_frame_offset
        logging.info("Resuming collector frame count from %s", resume_frame_offset)

    def _capture_policy_resume_state():
        state = {}
        for attr_name in (
            "actor_opt",
            "critic_opt",
            "actor_opt_scheduler",
            "critic_opt_scheduler",
        ):
            target = getattr(policy, attr_name, None)
            if target is None or not hasattr(target, "state_dict"):
                continue
            try:
                state[attr_name] = _recursive_to_cpu(target.state_dict())
            except Exception as e:
                logging.warning(
                    "Could not capture policy sidecar state for %s: %s",
                    attr_name,
                    e,
                )
        return state

    def _build_resume_bundle():
        bundle = {
            "version": 1,
            "policy": _capture_policy_resume_state(),
            "trainer": {
                "collector_frames": int(collector._frames),
                "controller_phase": int(controller_phase),
                "controller_speed_pressure": float(controller_speed_pressure),
                "controller_exploration_pressure": float(
                    controller_exploration_pressure
                ),
                "controller_collapse_active": bool(controller_collapse_active),
                "controller_completion_ema": float(controller_completion_ema),
                "controller_gate0_crash_ema": float(controller_gate0_crash_ema),
                "controller_completion_signal": float(controller_completion_signal),
                "controller_crash_signal": float(controller_crash_signal),
                "controller_t20_lap_time_sec": controller_t20_lap_time_sec,
                "controller_t20_best_stable_lap_time_sec": (
                    controller_t20_best_stable_lap_time_sec
                ),
                "controller_phase2_ready_count": int(controller_phase2_ready_count),
                "controller_plateau_count": int(controller_plateau_count),
                "controller_phase3_has_locked": bool(controller_phase3_has_locked),
                "controller_phase3_bad_window_count": int(
                    controller_phase3_bad_window_count
                ),
                "controller_phase_start_frames": int(controller_phase_start_frames),
                "controller_completion_history": list(controller_completion_history),
                "controller_completion_recent": list(controller_completion_recent),
                "controller_gate0_crash_recent": list(
                    controller_gate0_crash_recent
                ),
                "current_entropy_coef": float(current_entropy_coef),
                "target_entropy_coef": float(target_entropy_coef),
                "logged_curriculum_phase": int(logged_curriculum_phase),
                "phase2_started_env_frames": phase2_started_env_frames,
                "phase2_started_iter": phase2_started_iter,
                "phase2_started_accuracy_ema": phase2_started_accuracy_ema,
                "phase3_started_env_frames": phase3_started_env_frames,
                "phase3_started_iter": phase3_started_iter,
            },
        }
        if hasattr(base_env, "get_resume_state"):
            bundle["env"] = _recursive_to_cpu(base_env.get_resume_state())
        return _recursive_to_cpu(bundle)

    def _save_checkpoint_bundle(checkpoint_path: str):
        torch.save(policy.state_dict(), checkpoint_path)
        sidecar_path = _checkpoint_sidecar_path(checkpoint_path)
        torch.save(_build_resume_bundle(), sidecar_path)
        return sidecar_path

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
                if _should_log_wandb_stat_key(("stats", k))
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
    phase2_plateau_floor = 0.0
    phase3_rebound_cap_active = False
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
            if hasattr(policy, "entropy_coef"):
                info["entropy/current_coef"] = current_entropy_coef
                info["entropy/target_coef"] = target_entropy_coef
                info["entropy/phase2_plateau_floor"] = phase2_plateau_floor
                info["entropy/phase3_lock_active"] = float(controller_phase >= 3)
                info["entropy/phase3_rebound_cap_active"] = float(
                    phase3_rebound_cap_active
                )
            info["controller/phase"] = float(controller_phase)
            info["controller/speed_pressure"] = controller_speed_pressure
            info["controller/exploration_pressure"] = controller_exploration_pressure
            info["controller/phase3_has_locked"] = float(
                controller_phase3_has_locked
            )
            info["controller/phase3_bad_window_count"] = float(
                controller_phase3_bad_window_count
            )
            episode_stats.add(data.to_tensordict())

            if len(episode_stats) >= base_env.num_envs:
                raw_stats = episode_stats.pop()
                stats = {
                    "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                    for k, v in raw_stats.items(True, True)
                    if _should_log_wandb_stat_key(k)
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

                def _masked_min(key, mask):
                    values = _stats_tensor(key)
                    if values is None or mask is None or not bool(mask.any().item()):
                        return None
                    return values.float()[mask].min().item()

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
                lap_time_all = _mean("lap_time_steps")  # 0 for non-completions in raw stats

                task_mean_speed = _masked_mean("mean_speed", gate0_start_mask)
                task_max_speed = _masked_mean("max_speed", gate0_start_mask)
                task_gates_passed = _masked_mean("gates_passed", gate0_start_mask)
                task_success_rate = _masked_mean("success", gate0_start_mask)
                task_ep_len = _masked_mean("episode_len", gate0_start_mask)
                task_completion_time_steps = _masked_mean("lap_time_steps", gate0_success_mask)
                task_completion_time_steps_fastest = _masked_min("lap_time_steps", gate0_success_mask)
                task_success_mean_speed = _masked_mean("mean_speed", gate0_success_mask)
                task_success_peak_speed = _masked_mean("max_speed", gate0_success_mask)

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
                rew_gate_reentry = _mean("reward_gate_reentry")
                rew_exit_anchor  = _mean("reward_exit_anchor")
                rew_guided_exit  = _mean("reward_guided_exit")
                curriculum_phase = _mean("curriculum_phase")
                curriculum_accuracy = _mean("curriculum_accuracy_ema")
                curriculum_furthest = _mean("curriculum_furthest_ema")
                reset_active_gate = _mean("reset_active_gate")
                reset_from_gate0 = _mean("reset_from_gate0")
                reset_from_prev_gate = _mean("reset_from_prev_gate")
                reset_from_random_gate = _mean("reset_from_random_gate")
                cheating_events = _mean("cheating")
                cheating_values = _stats_tensor("cheating")
                cheating_rate = None
                if cheating_values is not None:
                    cheating_rate = cheating_values.gt(0).float().mean().item()
                reentry_events = _mean("gate_reentry")
                reentry_values = _stats_tensor("gate_reentry")
                reentry_rate = None
                if reentry_values is not None:
                    reentry_rate = reentry_values.gt(0).float().mean().item()
                guided_exit_events = _mean("guided_exit_failures")
                guided_exit_values = _stats_tensor("guided_exit_failures")
                guided_exit_rate = None
                if guided_exit_values is not None:
                    guided_exit_rate = guided_exit_values.gt(0).float().mean().item()
                gate_12_hard_reentry_failures = _mean("gate_12_hard_reentry_failures")
                gate_12_commit_timeout_failures = _mean("gate_12_commit_timeout_failures")
                gate_12_commit_successes = _mean("gate_12_commit_successes")
                gate_12_commit_attempts = _mean("gate_12_commit_attempts")
                gate_12_commit_success_rate = None
                if (
                    gate_12_commit_successes is not None
                    and gate_12_commit_attempts is not None
                    and gate_12_commit_attempts > 0
                ):
                    gate_12_commit_success_rate = (
                        gate_12_commit_successes
                        / max(gate_12_commit_attempts, 1e-6)
                    )

                derived = {}

                # Racing metrics
                if mean_speed   is not None: derived["race/mean_speed_ms"]       = mean_speed
                if max_speed    is not None: derived["race/max_speed_ms"]         = max_speed
                if gates_passed is not None: derived["race/gates_passed_per_ep"]  = gates_passed
                if success_rate is not None: derived["race/lap_completion_rate"]  = success_rate
                if ep_len       is not None: derived["race/episode_len_steps"]    = ep_len
                if ep_len       is not None: derived["race/episode_len_sec"]      = ep_len * cfg.sim.dt * cfg.sim.substeps
                # Simple aliases for W&B dashboards that expect the original names.
                if mean_speed   is not None: derived["simple/mean_speed_ms"] = mean_speed
                if max_speed    is not None: derived["simple/max_speed_ms"] = max_speed
                if gates_passed is not None: derived["simple/gates_passed"] = gates_passed
                if ep_len       is not None: derived["simple/episode_time_sec"] = ep_len * cfg.sim.dt * cfg.sim.substeps
                derived["simple/full_course_completion_rate"] = 0.0
                derived["simple/full_course_completion_pct"] = 0.0
                derived["simple/full_course_completion_time_steps_mean"] = 0.0
                derived["simple/full_course_completion_time_sec_mean"] = 0.0
                derived["simple/full_course_completion_time_steps_fastest"] = 0.0
                derived["simple/full_course_completion_time_sec_fastest"] = 0.0
                derived["simple/full_course_completion_mean_speed_ms"] = 0.0
                derived["simple/full_course_completion_peak_speed_ms"] = 0.0
                derived["race_objective/target_completion_rate"] = (
                    race_objective_completion_target
                )
                derived["race_objective/target_lap_time_sec"] = (
                    race_objective_lap_time_target_sec
                )
                effective_lap_time = lap_time
                if effective_lap_time is None and success_rate and success_rate > 0:
                    effective_lap_time = lap_time_all
                if effective_lap_time is not None:
                    derived["race/lap_time_steps"] = effective_lap_time
                    derived["race/lap_time_sec"] = effective_lap_time * cfg.sim.dt * cfg.sim.substeps
                if gate0_episode_count > 0:
                    if task_mean_speed is not None:
                        derived["simple/mean_speed_ms"] = task_mean_speed
                    if task_max_speed is not None:
                        derived["simple/max_speed_ms"] = task_max_speed
                    if task_gates_passed is not None:
                        derived["simple/gates_passed"] = task_gates_passed
                    if task_success_rate is not None:
                        derived["simple/full_course_completion_rate"] = task_success_rate
                        derived["simple/full_course_completion_pct"] = 100.0 * task_success_rate
                        derived["race_objective/full_lap_completion_rate"] = (
                            task_success_rate
                        )
                        derived["race_objective/full_lap_completion_margin"] = (
                            task_success_rate - race_objective_completion_target
                        )
                        derived["race_objective/full_lap_completion_goal_met"] = float(
                            task_success_rate >= race_objective_completion_target
                        )
                    if task_ep_len is not None:
                        derived["simple/episode_time_sec"] = task_ep_len * cfg.sim.dt * cfg.sim.substeps
                if gate0_success_count > 0 and task_completion_time_steps is not None:
                    completion_time_sec_mean = task_completion_time_steps * cfg.sim.dt * cfg.sim.substeps
                    derived["simple/completion_time_steps_mean"] = task_completion_time_steps
                    derived["simple/completion_time_sec_mean"] = completion_time_sec_mean
                    derived["simple/full_course_completion_time_steps_mean"] = task_completion_time_steps
                    derived["simple/full_course_completion_time_sec_mean"] = completion_time_sec_mean
                    derived["race_objective/full_lap_time_sec_mean"] = (
                        completion_time_sec_mean
                    )
                    derived["race_objective/full_lap_time_margin_to_13s"] = (
                        race_objective_lap_time_target_sec - completion_time_sec_mean
                    )
                    derived["race_objective/full_lap_time_goal_met"] = float(
                        completion_time_sec_mean <= race_objective_lap_time_target_sec
                    )
                if gate0_success_count > 0 and task_completion_time_steps_fastest is not None:
                    completion_time_sec_fastest = (
                        task_completion_time_steps_fastest * cfg.sim.dt * cfg.sim.substeps
                    )
                    derived["simple/full_course_completion_time_steps_fastest"] = (
                        task_completion_time_steps_fastest
                    )
                    derived["simple/full_course_completion_time_sec_fastest"] = (
                        completion_time_sec_fastest
                    )
                    derived["race_objective/full_lap_time_sec_fastest"] = (
                        completion_time_sec_fastest
                    )
                if gate0_success_count > 0 and task_success_mean_speed is not None:
                    derived["simple/full_course_completion_mean_speed_ms"] = task_success_mean_speed
                if gate0_success_count > 0 and task_success_peak_speed is not None:
                    derived["simple/full_course_completion_peak_speed_ms"] = task_success_peak_speed
                if (
                    "race_objective/full_lap_completion_rate" in derived
                    and "race_objective/full_lap_time_sec_mean" in derived
                ):
                    derived["race_objective/goal_met"] = float(
                        derived["race_objective/full_lap_completion_rate"]
                        >= race_objective_completion_target
                        and derived["race_objective/full_lap_time_sec_mean"]
                        <= race_objective_lap_time_target_sec
                    )
                if cheating_rate is not None:
                    derived["cheating/episode_rate"] = cheating_rate
                if cheating_events is not None:
                    derived["cheating/events_per_episode"] = cheating_events
                if reentry_rate is not None:
                    derived["simple/gate_reentry"] = reentry_rate
                    derived["reentry/episode_rate"] = reentry_rate
                if reentry_events is not None:
                    derived["reentry/events_per_episode"] = reentry_events
                if guided_exit_rate is not None:
                    derived["guided_exit/episode_rate"] = guided_exit_rate
                if guided_exit_events is not None:
                    derived["guided_exit/events_per_episode"] = guided_exit_events
                    derived["fixes/guided_exit_failures"] = guided_exit_events
                if gate_12_hard_reentry_failures is not None:
                    derived["fixes/gate_12_hard_reentry_failures"] = gate_12_hard_reentry_failures
                    derived["fixes/human_gate_12_hard_reentry_failures"] = gate_12_hard_reentry_failures
                if gate_12_commit_timeout_failures is not None:
                    derived["fixes/gate_12_commit_timeout_failures"] = gate_12_commit_timeout_failures
                    derived["fixes/human_gate_12_commit_timeout_failures"] = gate_12_commit_timeout_failures
                if gate_12_commit_success_rate is not None:
                    derived["fixes/gate_12_commit_success_rate"] = gate_12_commit_success_rate
                    derived["fixes/human_gate_12_commit_success_rate"] = gate_12_commit_success_rate
                if rew_gate_reentry is not None:
                    derived["reentry/reward_gate_reentry"] = rew_gate_reentry
                    derived["fixes/reward_gate_reentry"] = rew_gate_reentry
                if rew_exit_anchor is not None:
                    derived["reentry/reward_exit_anchor"] = rew_exit_anchor
                    derived["fixes/reward_exit_anchor"] = rew_exit_anchor
                if rew_guided_exit is not None:
                    derived["guided_exit/reward_guided_exit"] = rew_guided_exit
                    derived["fixes/reward_guided_exit"] = rew_guided_exit
                if furthest_gate is not None:
                    derived["race/furthest_gate_reached"] = furthest_gate
                    derived["race/furthest_gate_number"] = furthest_gate + 1.0
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
                if rew_guided_exit is not None: derived["reward/guided_exit_cumul"] = rew_guided_exit
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
                    if rew_guided_exit is not None: derived["reward/guided_exit_fraction"] = rew_guided_exit / rew_return
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
                phase2_accuracy_threshold = float(
                    cfg.task.get("phase2_speed_focus_accuracy_rate", 0.40)
                )
                current_phase = int(controller_phase)
                phase2_active = current_phase >= 2
                phase2_just_unlocked = logged_curriculum_phase < 2 <= current_phase
                phase3_just_locked = logged_curriculum_phase < 3 <= current_phase
                derived["curriculum/phase2_active"] = float(phase2_active)
                derived["curriculum/phase2_just_unlocked"] = float(phase2_just_unlocked)
                if curriculum_accuracy is not None:
                    derived["curriculum/phase2_accuracy_gate_threshold"] = (
                        phase2_accuracy_threshold
                    )
                    derived["curriculum/phase2_accuracy_gate_met"] = float(
                        curriculum_accuracy >= phase2_accuracy_threshold
                    )
                    derived["curriculum/phase2_accuracy_gate_margin"] = (
                        curriculum_accuracy - phase2_accuracy_threshold
                    )
                if phase2_just_unlocked and phase2_started_env_frames is None:
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
                        "accuracy_ema=%s after the controller unlock rule was satisfied",
                        phase2_started_env_frames,
                        phase2_started_iter,
                        phase2_started_accuracy_ema,
                    )
                if phase3_just_locked and phase3_started_env_frames is None:
                    phase3_started_env_frames = int(collector._frames)
                    phase3_started_iter = int(i)
                    run.summary["curriculum/phase3_started_env_frames"] = (
                        phase3_started_env_frames
                    )
                    run.summary["curriculum/phase3_started_iter"] = phase3_started_iter
                    logging.info(
                        "Curriculum entered phase 3 lock-in at env_frames=%s iteration=%s",
                        phase3_started_env_frames,
                        phase3_started_iter,
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
                if phase3_started_env_frames is not None:
                    derived["curriculum/phase3_started_env_frames"] = float(
                        phase3_started_env_frames
                    )
                    derived["curriculum/phase3_started_iter"] = float(
                        phase3_started_iter
                    )
                logged_curriculum_phase = current_phase
                speed_phase_scale = 0.0
                if curriculum_phase is not None:
                    if curriculum_phase >= 2.0:
                        speed_phase_scale = cfg.task.get("reward_speed_scale_phase2", cfg.task.get("reward_speed_scale", 0.0))
                    elif curriculum_phase >= 1.0:
                        speed_phase_scale = cfg.task.get("reward_speed_scale", 0.0)
                derived["reward/speed_phase_scale"] = speed_phase_scale
                if reset_active_gate is not None:
                    derived["curriculum/reset_active_gate"] = reset_active_gate
                    if reset_active_gate >= 0:
                        derived["curriculum/reset_active_gate_number"] = reset_active_gate + 1.0
                if reset_from_gate0 is not None:
                    derived["curriculum/reset_from_gate0_rate"] = reset_from_gate0
                if reset_from_prev_gate is not None:
                    derived["curriculum/reset_from_prev_gate_rate"] = reset_from_prev_gate
                if reset_from_random_gate is not None:
                    derived["curriculum/reset_from_random_rate"] = reset_from_random_gate
                reset_gate_pass_emas = {}
                for gi in range(required_gate_count):
                    reset_gate_ema = _mean(f"reset_gate_{gi}_pass_ema")
                    if reset_gate_ema is not None:
                        reset_gate_pass_emas[gi] = float(reset_gate_ema)
                        derived[f"curriculum/reset_gate_{gi:02d}_pass_ema"] = reset_gate_ema
                        derived[f"curriculum_human/reset_gate_{gi + 1:02d}_pass_ema"] = reset_gate_ema
                derived["curriculum/min_accuracy_phase_frames"] = cfg.task.get(
                    "curriculum_min_phase_frames", 2_000_000
                )
                derived["curriculum/speed_unlock_accuracy_rate"] = cfg.task.get(
                    "phase_speed_unlock_accuracy_rate", 0.40
                )
                derived["curriculum/ang_decay_end_frames"] = cfg.task.get(
                    "angular_penalty_decay_frames", 100_000_000
                )

                step_sec = cfg.sim.dt * cfg.sim.substeps
                gate0_crash_rate_batch = _masked_mean("collision", gate0_start_mask)
                if gate0_episode_count > 0 and task_success_rate is not None:
                    batch_completion_rate = float(task_success_rate)
                    if not controller_completion_recent:
                        controller_completion_ema = batch_completion_rate
                    else:
                        controller_completion_ema = (
                            (1.0 - controller_ema_alpha) * controller_completion_ema
                            + controller_ema_alpha * batch_completion_rate
                        )
                    controller_completion_recent.append(batch_completion_rate)
                if gate0_episode_count > 0 and gate0_crash_rate_batch is not None:
                    batch_gate0_crash_rate = float(gate0_crash_rate_batch)
                    if not controller_gate0_crash_recent:
                        controller_gate0_crash_ema = batch_gate0_crash_rate
                    else:
                        controller_gate0_crash_ema = (
                            (1.0 - controller_ema_alpha) * controller_gate0_crash_ema
                            + controller_ema_alpha * batch_gate0_crash_rate
                        )
                    controller_gate0_crash_recent.append(batch_gate0_crash_rate)

                if controller_completion_recent:
                    controller_completion_signal = float(
                        sum(controller_completion_recent) / len(controller_completion_recent)
                    )
                else:
                    controller_completion_signal = controller_completion_ema

                if controller_gate0_crash_recent:
                    controller_crash_signal = float(
                        sum(controller_gate0_crash_recent) / len(controller_gate0_crash_recent)
                    )
                else:
                    controller_crash_signal = controller_gate0_crash_ema

                controller_completion_history.append(controller_completion_signal)
                completion_drop_5 = 0.0
                if len(controller_completion_history) >= controller_completion_history.maxlen:
                    completion_drop_5 = (
                        controller_completion_signal - controller_completion_history[0]
                    )

                hard_gate_reliability = 0.0
                if getattr(base_env, "controller_hard_gate_indices", ()):
                    hard_gate_values = [
                        reset_gate_pass_emas[gi]
                        for gi in base_env.controller_hard_gate_indices
                        if gi in reset_gate_pass_emas
                    ]
                    if hard_gate_values:
                        hard_gate_reliability = min(hard_gate_values)

                final_gate_reliability = 0.0
                if getattr(base_env, "controller_final_gate_indices", ()):
                    final_gate_values = [
                        reset_gate_pass_emas[gi]
                        for gi in base_env.controller_final_gate_indices
                        if gi in reset_gate_pass_emas
                    ]
                    if final_gate_values:
                        final_gate_reliability = min(final_gate_values)

                t20_valid = False
                lap_time_steps_values = _stats_tensor("lap_time_steps")
                if (
                    lap_time_steps_values is not None
                    and gate0_success_mask is not None
                    and gate0_success_count >= controller_success_min_for_t20
                ):
                    gate0_success_lap_times = lap_time_steps_values.float()[gate0_success_mask]
                    controller_t20_lap_time_sec = (
                        torch.quantile(gate0_success_lap_times, 0.20).item() * step_sec
                    )
                    t20_valid = True

                stabilize_window = (
                    controller_completion_signal >= controller_stabilize_completion_target
                    and hard_gate_reliability >= controller_stabilize_hard_gate_target
                    and final_gate_reliability >= controller_stabilize_final_gate_target
                    and controller_crash_signal <= controller_stabilize_crash_target
                )
                phase2_operating_window = (
                    controller_phase >= 2
                    and controller_completion_signal >= controller_phase2_completion_target
                    and hard_gate_reliability >= controller_phase2_hard_gate_target
                    and final_gate_reliability >= controller_phase2_final_gate_target
                    and controller_crash_signal <= controller_phase2_crash_target
                )
                phase3_lock_window = (
                    controller_phase == 2
                    and controller_speed_pressure >= controller_phase3_lock_speed_pressure
                    and controller_completion_signal
                    >= controller_phase3_lock_completion_target
                    and final_gate_reliability
                    >= controller_phase3_lock_final_gate_target
                    and controller_crash_signal <= controller_phase3_lock_crash_target
                )

                if (stabilize_window or phase2_operating_window or phase3_lock_window) and t20_valid:
                    if controller_t20_best_stable_lap_time_sec is None:
                        controller_t20_best_stable_lap_time_sec = controller_t20_lap_time_sec
                        controller_plateau_count = 0
                    else:
                        best_before_update = controller_t20_best_stable_lap_time_sec
                        relative_improvement = (
                            (best_before_update - controller_t20_lap_time_sec)
                            / max(best_before_update, 1e-6)
                        )
                        controller_t20_best_stable_lap_time_sec = min(
                            controller_t20_best_stable_lap_time_sec,
                            controller_t20_lap_time_sec,
                        )
                        if relative_improvement >= controller_plateau_improvement_eps:
                            controller_plateau_count = 0
                        else:
                            controller_plateau_count += 1
                else:
                    controller_plateau_count = 0

                phase1_collapse_conditions = (
                    controller_completion_signal < controller_completion_recover
                    or final_gate_reliability < controller_completion_recover
                    or controller_crash_signal > controller_crash_recover
                    or completion_drop_5 < -controller_completion_drop_recover
                )
                phase2_collapse_conditions = (
                    controller_completion_signal < controller_phase2_completion_floor
                    or final_gate_reliability < controller_phase2_final_gate_floor
                    or controller_crash_signal > controller_phase2_crash_recover
                    or completion_drop_5 < -controller_completion_drop_recover
                )
                phase3_fallback_conditions = (
                    controller_completion_signal < controller_phase3_completion_floor
                    or final_gate_reliability < controller_phase3_final_gate_floor
                    or controller_crash_signal > controller_phase3_crash_recover
                    or completion_drop_5 < -controller_phase3_completion_drop_recover
                )
                if controller_phase >= 3:
                    if phase3_fallback_conditions:
                        controller_phase3_bad_window_count += 1
                    else:
                        controller_phase3_bad_window_count = 0
                else:
                    controller_phase3_bad_window_count = 0
                phase1_unlock = (
                    gate0_success_count > 0 or controller_completion_signal >= 0.35
                )
                if controller_phase == 0 and phase1_unlock:
                    controller_phase = 1
                    controller_phase_start_frames = collector._frames
                    controller_phase2_ready_count = 0
                    controller_plateau_count = 0
                    controller_speed_pressure = max(
                        controller_speed_pressure,
                        controller_phase1_initial_speed_pressure,
                    )

                phase3_lock_conditions = (
                    phase3_lock_window
                    and controller_plateau_count >= controller_phase3_lock_plateau_windows
                    and t20_valid
                )
                phase3_fallback_ready = (
                    controller_phase >= 3
                    and phase3_fallback_conditions
                    and controller_phase3_bad_window_count
                    >= controller_phase3_fallback_patience_windows
                )
                if phase3_fallback_ready:
                    controller_phase = 2
                    controller_phase_start_frames = collector._frames
                    controller_phase2_ready_count = 0
                    controller_plateau_count = 0
                    controller_phase3_bad_window_count = 0
                    controller_exploration_pressure = 0.0
                elif phase3_lock_conditions:
                    controller_phase = 3
                    controller_phase_start_frames = collector._frames
                    controller_phase2_ready_count = 0
                    controller_phase3_has_locked = True
                    controller_phase3_bad_window_count = 0
                    controller_exploration_pressure = 0.0
                    target_entropy_coef = phase3_lock_coef
                    if phase3_started_env_frames is None:
                        phase3_started_env_frames = int(collector._frames)
                        phase3_started_iter = int(i)
                        run.summary["curriculum/phase3_started_env_frames"] = (
                            phase3_started_env_frames
                        )
                        run.summary["curriculum/phase3_started_iter"] = (
                            phase3_started_iter
                        )
                        logging.info(
                            "Entering phase 3 lock-in at env_frames=%s iteration=%s",
                            phase3_started_env_frames,
                            phase3_started_iter,
                        )
                elif controller_phase == 2 and phase2_collapse_conditions:
                    controller_phase_start_frames = collector._frames
                    controller_phase2_ready_count = 0
                    controller_plateau_count = 0
                    controller_exploration_pressure = 0.0
                    if not (
                        controller_phase3_sticky_after_first_lock
                        and controller_phase3_has_locked
                    ):
                        controller_phase = 1

                phase_frames = max(collector._frames - controller_phase_start_frames, 0)
                phase2_ready_window = (
                    controller_phase == 1
                    and stabilize_window
                    and phase_frames >= controller_min_phase_frames
                )
                if phase2_ready_window:
                    controller_phase2_ready_count += 1
                elif controller_phase == 1:
                    controller_phase2_ready_count = 0

                if (
                    controller_phase == 1
                    and controller_phase2_ready_count >= controller_plateau_windows
                ):
                    controller_phase = 2
                    controller_phase_start_frames = collector._frames
                    controller_phase2_ready_count = 0
                    controller_plateau_count = 0
                    controller_speed_pressure = max(
                        controller_speed_pressure,
                        controller_phase2_initial_speed_pressure,
                    )
                    if phase2_started_env_frames is None:
                        phase2_started_env_frames = int(collector._frames)
                        phase2_started_iter = int(i)
                        phase2_started_accuracy_ema = (
                            float(curriculum_accuracy)
                            if curriculum_accuracy is not None
                            else None
                        )
                        run.summary["curriculum/phase2_started_env_frames"] = (
                            phase2_started_env_frames
                        )
                        run.summary["curriculum/phase2_started_iter"] = (
                            phase2_started_iter
                        )
                        if phase2_started_accuracy_ema is not None:
                            run.summary["curriculum/phase2_started_accuracy_ema"] = (
                                phase2_started_accuracy_ema
                            )
                        logging.info(
                            "Curriculum entered phase 2 at env_frames=%s iteration=%s "
                            "accuracy_ema=%s after the controller unlock rule was satisfied",
                            phase2_started_env_frames,
                            phase2_started_iter,
                            phase2_started_accuracy_ema,
                        )

                phase2_operating_window = (
                    controller_phase >= 2
                    and controller_completion_signal >= controller_phase2_completion_target
                    and hard_gate_reliability >= controller_phase2_hard_gate_target
                    and final_gate_reliability >= controller_phase2_final_gate_target
                    and controller_crash_signal <= controller_phase2_crash_target
                )
                if controller_phase >= 3:
                    controller_collapse_active = phase3_fallback_conditions
                    slack = min(
                        controller_completion_signal
                        - controller_phase3_lock_completion_target,
                        final_gate_reliability
                        - controller_phase3_lock_final_gate_target,
                        controller_phase3_lock_crash_target
                        - controller_crash_signal,
                    )
                elif controller_phase >= 2:
                    controller_collapse_active = phase2_collapse_conditions
                    slack = min(
                        controller_completion_signal - controller_phase2_completion_target,
                        hard_gate_reliability - controller_phase2_hard_gate_target,
                        final_gate_reliability - controller_phase2_final_gate_target,
                        controller_phase2_crash_target - controller_crash_signal,
                    )
                else:
                    controller_collapse_active = (
                        controller_phase == 1
                        and controller_speed_pressure > controller_phase1_initial_speed_pressure + 1e-6
                        and phase1_collapse_conditions
                    )
                    slack = min(
                        controller_completion_signal
                        - controller_stabilize_completion_target,
                        hard_gate_reliability
                        - controller_stabilize_hard_gate_target,
                        final_gate_reliability
                        - controller_stabilize_final_gate_target,
                        controller_stabilize_crash_target
                        - controller_crash_signal,
                    )

                if controller_phase < 1:
                    controller_speed_pressure = 0.0
                elif controller_phase == 1:
                    updated_speed_pressure = max(
                        controller_speed_pressure,
                        controller_phase1_initial_speed_pressure,
                    )
                    if stabilize_window:
                        updated_speed_pressure += controller_speed_up_step
                    elif (
                        controller_completion_signal < controller_completion_floor
                        or final_gate_reliability < controller_completion_floor
                        or controller_crash_signal > controller_crash_recover
                    ):
                        updated_speed_pressure -= controller_speed_down_step
                    if controller_collapse_active:
                        updated_speed_pressure -= controller_speed_panic_step
                    controller_speed_pressure = _clamp01(updated_speed_pressure)
                elif controller_phase == 2:
                    updated_speed_pressure = max(
                        controller_speed_pressure,
                        controller_phase2_initial_speed_pressure,
                    )
                    if phase2_operating_window:
                        updated_speed_pressure += controller_phase2_speed_up_step
                    elif slack < 0.0:
                        updated_speed_pressure -= controller_phase2_speed_down_step
                    if controller_collapse_active:
                        updated_speed_pressure -= controller_phase2_panic_step
                    controller_speed_pressure = _clamp01(updated_speed_pressure)
                else:
                    controller_speed_pressure = _clamp01(controller_speed_pressure)

                plateau_active = (
                    controller_phase == 2
                    and phase2_operating_window
                    and t20_valid
                    and controller_t20_best_stable_lap_time_sec is not None
                    and controller_plateau_count >= controller_plateau_windows
                    and controller_speed_pressure >= controller_phase2_plateau_start_pressure
                )
                exploration_target = 0.0
                phase2_plateau_floor = 0.0
                if controller_phase >= 3:
                    controller_exploration_pressure = 0.0
                else:
                    if plateau_active:
                        exploration_target = _clamp01(slack / 0.08)
                    controller_exploration_pressure = _clamp01(
                        _slew_toward_asymmetric(
                            controller_exploration_pressure,
                            exploration_target,
                            max_increase=0.05,
                            max_decrease=0.15,
                        )
                    )

                if entropy_controller_enabled:
                    phase3_rebound_cap_active = False
                    entropy_max_increase = entropy_increase_step
                    entropy_max_decrease = entropy_decrease_step
                    if controller_phase >= 3:
                        target_entropy_coef = phase3_lock_coef
                        entropy_max_increase = phase3_max_increase_step
                        entropy_max_decrease = phase3_max_decrease_step
                    elif controller_completion_ema < 0.20:
                        target_entropy_coef = phase0_coef_max
                    elif controller_collapse_active:
                        target_entropy_coef = recovery_coef
                    else:
                        entropy_floor = (
                            phase2_min_coef if controller_phase >= 2 else phase1_min_coef
                        )
                        target_entropy_coef = entropy_floor + (
                            1.0 - controller_speed_pressure
                        ) * (phase0_coef_max - entropy_floor)
                        if plateau_active:
                            phase2_plateau_floor = min(
                                phase2_plateau_max_coef,
                                phase2_plateau_base_coef
                                + phase2_plateau_gain
                                * controller_exploration_pressure,
                            )
                            target_entropy_coef = max(
                                target_entropy_coef,
                                phase2_plateau_floor,
                            )
                            entropy_max_increase = min(
                                entropy_max_increase,
                                phase2_plateau_max_increase_step,
                            )
                    # Once phase 3 has locked at least once, keep recovery search mild.
                    if controller_phase3_has_locked:
                        current_entropy_coef = min(
                            current_entropy_coef,
                            phase3_rebound_cap,
                        )
                        clamped_target_entropy_coef = min(
                            target_entropy_coef,
                            phase3_rebound_cap,
                        )
                        phase3_rebound_cap_active = (
                            clamped_target_entropy_coef + 1e-12 < target_entropy_coef
                        )
                        target_entropy_coef = clamped_target_entropy_coef
                    current_entropy_coef = _slew_toward_asymmetric(
                        current_entropy_coef,
                        target_entropy_coef,
                        max_increase=entropy_max_increase,
                        max_decrease=entropy_max_decrease,
                    )
                    policy.entropy_coef = current_entropy_coef

                base_env.set_constrained_speed_controller(
                    phase=controller_phase,
                    speed_pressure=controller_speed_pressure,
                    exploration_pressure=controller_exploration_pressure,
                    collapse_active=controller_collapse_active,
                    entropy_target=target_entropy_coef,
                )

                derived["curriculum/phase"] = float(controller_phase)
                derived["controller/phase"] = float(controller_phase)
                derived["controller/slack"] = slack
                derived["controller/gate0_completion_ema"] = controller_completion_ema
                derived["controller/gate0_completion_signal"] = controller_completion_signal
                derived["controller/hard_gate_reliability"] = hard_gate_reliability
                derived["controller/final_gate_reliability"] = final_gate_reliability
                derived["controller/crash_ema_gate0"] = controller_gate0_crash_ema
                derived["controller/crash_signal_gate0"] = controller_crash_signal
                derived["controller/completion_drop_5"] = completion_drop_5
                derived["controller/speed_pressure"] = controller_speed_pressure
                derived["controller/exploration_pressure"] = controller_exploration_pressure
                derived["controller/exploration_episode_rate"] = float(
                    getattr(base_env, "controller_explore_episode_rate", 0.0)
                )
                derived["controller/collapse_active"] = float(controller_collapse_active)
                derived["controller/phase3_has_locked"] = float(
                    controller_phase3_has_locked
                )
                derived["controller/phase3_bad_window_count"] = float(
                    controller_phase3_bad_window_count
                )
                derived["controller/stabilize_window"] = float(stabilize_window)
                derived["controller/phase2_ready_window"] = float(phase2_ready_window)
                derived["controller/phase2_operating_window"] = float(phase2_operating_window)
                derived["controller/phase2_ready_windows"] = float(
                    controller_phase2_ready_count
                )
                derived["controller/plateau_count"] = float(controller_plateau_count)
                derived["controller/plateau_windows"] = float(controller_plateau_windows)
                derived["controller/min_phase_frames"] = float(controller_min_phase_frames)
                derived["controller/phase3_fallback_patience_windows"] = float(
                    controller_phase3_fallback_patience_windows
                )
                derived["controller/stabilize_completion_target"] = (
                    controller_stabilize_completion_target
                )
                derived["controller/stabilize_hard_gate_target"] = (
                    controller_stabilize_hard_gate_target
                )
                derived["controller/stabilize_final_gate_target"] = (
                    controller_stabilize_final_gate_target
                )
                derived["controller/stabilize_crash_target"] = (
                    controller_stabilize_crash_target
                )
                derived["controller/phase2_completion_target"] = (
                    controller_phase2_completion_target
                )
                derived["controller/phase2_hard_gate_target"] = (
                    controller_phase2_hard_gate_target
                )
                derived["controller/phase2_final_gate_target"] = (
                    controller_phase2_final_gate_target
                )
                derived["controller/phase2_completion_floor"] = (
                    controller_phase2_completion_floor
                )
                derived["controller/phase2_crash_target"] = (
                    controller_phase2_crash_target
                )
                derived["curriculum/phase2_trigger_met"] = float(
                    controller_phase >= 2
                )
                derived["curriculum/phase2_trigger_threshold"] = float(
                    controller_plateau_windows
                )
                derived["curriculum/phase2_trigger_ready_window"] = float(
                    phase2_ready_window
                )
                derived["curriculum/phase2_trigger_ready_windows"] = float(
                    controller_phase2_ready_count
                )
                if controller_t20_lap_time_sec is not None:
                    derived["controller/t20_lap_time_sec"] = controller_t20_lap_time_sec
                    derived["race_objective/t20_lap_time_sec"] = (
                        controller_t20_lap_time_sec
                    )
                    derived["race_objective/t20_lap_time_margin_to_13s"] = (
                        race_objective_lap_time_target_sec
                        - controller_t20_lap_time_sec
                    )
                    derived["race_objective/t20_goal_met"] = float(
                        controller_t20_lap_time_sec
                        <= race_objective_lap_time_target_sec
                    )
                if controller_t20_best_stable_lap_time_sec is not None:
                    derived["controller/t20_best_stable_lap_time_sec"] = (
                        controller_t20_best_stable_lap_time_sec
                    )

                speed_phase_scale = 0.0
                if controller_phase >= 2:
                    speed_phase_scale = cfg.task.get(
                        "reward_speed_scale_phase2",
                        cfg.task.get("reward_speed_scale", 0.0),
                    )
                elif controller_phase >= 1:
                    speed_phase_scale = cfg.task.get("reward_speed_scale", 0.0)
                derived["reward/speed_phase_scale"] = (
                    speed_phase_scale * controller_speed_pressure
                )

                if hasattr(policy, "entropy_coef"):
                    derived["entropy/current_coef"] = current_entropy_coef
                    derived["entropy/target_coef"] = target_entropy_coef
                    derived["entropy/phase"] = float(controller_phase)
                    derived["entropy/speed_pressure"] = controller_speed_pressure
                    derived["entropy/exploration_pressure"] = controller_exploration_pressure
                    derived["entropy/phase2_plateau_floor"] = phase2_plateau_floor
                    derived["entropy/phase3_rebound_cap"] = phase3_rebound_cap
                    derived["entropy/phase3_rebound_cap_active"] = float(
                        phase3_rebound_cap_active
                    )
                    derived["entropy/phase3_lock_active"] = float(
                        controller_phase >= 3
                    )

                controller_window_metrics = base_env.consume_controller_window_metrics()
                controller_gate_count = int(getattr(base_env, "num_course_gates", required_gate_count))
                for gi in range(controller_gate_count):
                    derived[f"controller/gate_{gi:02d}_speed_target"] = (
                        controller_window_metrics["gate_speed_target_ms"][gi]
                    )
                    derived[f"controller/gate_{gi:02d}_overspeed_rate"] = (
                        controller_window_metrics["gate_overspeed_rate"][gi]
                    )
                    derived[f"controller/gate_{gi:02d}_split_time_sec"] = (
                        controller_window_metrics["gate_split_time_steps"][gi] * step_sec
                    )
                    derived[f"controller/gate_{gi:02d}_exit_speed_ms"] = (
                        controller_window_metrics["gate_exit_speed_ms"][gi]
                    )

                logged_gate_count = max(
                    int(getattr(base_env, "num_logged_gates", controller_gate_count)),
                    controller_gate_count,
                )

                # Use a single consistent scalar naming scheme for per-gate
                # crossing rates.
                for gi in range(logged_gate_count):
                    v = _mean(f"gate_{gi}_crosses")
                    if v is not None:
                        derived[f"gates/gate_cross_{gi + 1}"] = v

                # Keep a compact per-gate crossing summary for quick visual
                # comparison in W&B.
                gate_counts = [
                    _mean(f"gate_{gi}_crosses") or 0.0 for gi in range(logged_gate_count)
                ]
                gate_labels = [f"G{gi + 1}" for gi in range(logged_gate_count)]
                gate_table = wandb.Table(
                    columns=["gate", "mean_crosses"],
                    data=[[label, val] for label, val in zip(gate_labels, gate_counts)],
                )
                derived["gates/crossing_distribution"] = wandb.plot.bar(
                    gate_table,
                    "gate",
                    "mean_crosses",
                    title="Gate Crossing Distribution",
                )

                for gi in range(logged_gate_count):
                    v = _mean(f"cheating_gate_{gi}")
                    if v is not None:
                        derived[f"cheating/gate_{gi:02d}_events"] = v
                        derived[f"cheating_human/gate_{gi + 1:02d}_events"] = v

                cheating_gate_counts = [
                    _mean(f"cheating_gate_{gi}") or 0.0
                    for gi in range(logged_gate_count)
                ]
                cheating_gate_table = wandb.Table(
                    columns=["gate", "repeat_events"],
                    data=[
                        [label, val]
                        for label, val in zip(gate_labels, cheating_gate_counts)
                    ],
                )
                derived["cheating/gate_distribution"] = wandb.plot.bar(
                    cheating_gate_table,
                    "gate",
                    "repeat_events",
                    title="Consecutive Same-Gate Repeat Events",
                )

                for gi in range(logged_gate_count):
                    v = _mean(f"gate_reentry_gate_{gi}")
                    if v is not None:
                        derived[f"reentry/gate_{gi:02d}_events"] = v
                        derived[f"reentry_human/gate_{gi + 1:02d}_events"] = v

                target_gate_aliases = {
                    4: "05",
                    5: "06",
                    11: "12",
                }
                target_gate_total = 0.0
                target_gate_total_seen = False
                for zero_based_gate_idx, human_gate_label in target_gate_aliases.items():
                    v = _mean(f"gate_reentry_gate_{zero_based_gate_idx}")
                    if v is None:
                        continue
                    target_gate_total += v
                    target_gate_total_seen = True
                    derived[f"fixes/gate_{human_gate_label}_reentry_events"] = v
                    derived[f"fixes/human_gate_{human_gate_label}_reentry_events"] = v
                if target_gate_total_seen:
                    derived["fixes/target_gate_reentry_events"] = target_gate_total

                reentry_gate_counts = [
                    _mean(f"gate_reentry_gate_{gi}") or 0.0
                    for gi in range(logged_gate_count)
                ]
                reentry_gate_table = wandb.Table(
                    columns=["gate", "reentry_events"],
                    data=[[label, val] for label, val in zip(gate_labels, reentry_gate_counts)],
                )
                derived["reentry/gate_distribution"] = wandb.plot.bar(
                    reentry_gate_table,
                    "gate",
                    "reentry_events",
                    title="Gate Re-entry Events",
                )

                for gi in range(logged_gate_count):
                    v = _mean(f"guided_exit_failure_gate_{gi}")
                    if v is not None:
                        derived[f"guided_exit/gate_{gi:02d}_failures"] = v
                        derived[f"guided_exit_human/gate_{gi + 1:02d}_failures"] = v

                guided_exit_target_aliases = {
                    11: "12",
                }
                for zero_based_gate_idx, human_gate_label in guided_exit_target_aliases.items():
                    v = _mean(f"guided_exit_failure_gate_{zero_based_gate_idx}")
                    if v is not None:
                        derived[f"fixes/gate_{human_gate_label}_guided_exit_failures"] = v
                        derived[f"fixes/human_gate_{human_gate_label}_guided_exit_failures"] = v
                    v = _mean(f"cheating_gate_{zero_based_gate_idx}")
                    if v is not None:
                        derived[f"fixes/gate_{human_gate_label}_cheating_events"] = v
                        derived[f"fixes/human_gate_{human_gate_label}_cheating_events"] = v
                if gate_12_hard_reentry_failures is not None:
                    derived["guided_exit_human/gate_12_hard_reentry_failures"] = gate_12_hard_reentry_failures
                if gate_12_commit_timeout_failures is not None:
                    derived["guided_exit_human/gate_12_commit_timeout_failures"] = gate_12_commit_timeout_failures
                if gate_12_commit_success_rate is not None:
                    derived["guided_exit_human/gate_12_commit_success_rate"] = gate_12_commit_success_rate

                guided_exit_gate_counts = [
                    _mean(f"guided_exit_failure_gate_{gi}") or 0.0
                    for gi in range(logged_gate_count)
                ]
                guided_exit_gate_table = wandb.Table(
                    columns=["gate", "guided_exit_failures"],
                    data=[
                        [label, val]
                        for label, val in zip(gate_labels, guided_exit_gate_counts)
                    ],
                )
                derived["guided_exit/gate_distribution"] = wandb.plot.bar(
                    guided_exit_gate_table,
                    "gate",
                    "guided_exit_failures",
                    title="Guided Exit Failures",
                )

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
                    sidecar_path = _save_checkpoint_bundle(ckpt_path)
                    logging.info(
                        "Saved checkpoint to %s and trainer state to %s",
                        str(ckpt_path),
                        str(sidecar_path),
                    )
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
            sidecar_path = _save_checkpoint_bundle(ckpt_path)

            model_artifact = wandb.Artifact(
                f"{cfg.task.name}-{cfg.algo.name.lower()}",
                type="model",
                description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
                metadata=dict(cfg))

            model_artifact.add_file(ckpt_path)
            model_artifact.add_file(sidecar_path)
            wandb.save(ckpt_path)
            wandb.save(sidecar_path)
            run.log_artifact(model_artifact)

            logging.info(
                "Saved checkpoint to %s and trainer state to %s",
                str(ckpt_path),
                str(sidecar_path),
            )
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
