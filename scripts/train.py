import logging
import os
import random
import signal
import time
import yaml
import traceback

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from torchrl.envs.utils import set_exploration_type, ExplorationType

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


def _flat_stat(stats, key):
    try:
        value = stats.get(key)
    except KeyError:
        return None
    if value is None:
        return None
    return value.float().reshape(-1)


def _rate(mask):
    if mask.numel() == 0:
        return None
    return mask.float().mean().item()


def summarize_drone_race_stats(stats):
    """Build stable W&B keys for DroneRace episode batches."""
    start_gate = _flat_stat(stats, "start_gate_index")
    gates_passed = _flat_stat(stats, "gates_passed")
    success = _flat_stat(stats, "success")
    if start_gate is None or gates_passed is None or success is None:
        return {}

    info = {}
    success_mask = success > 0.5
    full_course_mask = start_gate < 0.5
    full_success_mask = full_course_mask & success_mask

    for gate_idx in range(13):
        gate_stat = _flat_stat(stats, f"gate_{gate_idx + 1:02d}_passed")
        if gate_stat is None:
            continue
        eligible = start_gate <= gate_idx
        if eligible.any():
            info[f"gates/gate{gate_idx + 1:02d}_completion_rate"] = (
                gate_stat[eligible] > 0.5
            ).float().mean().item()

    if full_course_mask.any():
        info["simple/full_course_completion_rate"] = (
            full_success_mask.float().sum() / full_course_mask.float().sum()
        ).item()
    info["simple/full_course_success_count"] = full_success_mask.float().sum().item()
    info["simple/avg_gates_passed"] = gates_passed.mean().item()

    if full_success_mask.any():
        mean_speed = _flat_stat(stats, "mean_speed")
        max_speed = _flat_stat(stats, "max_speed")
        lap_time = _flat_stat(stats, "lap_time_sec")
        if mean_speed is not None:
            info["simple/mean_speed"] = mean_speed[full_success_mask].mean().item()
        if max_speed is not None:
            info["simple/max_speed"] = max_speed[full_success_mask].max().item()
        if lap_time is not None:
            info["simple/total_course_completion_time"] = lap_time[full_success_mask].mean().item()
            info["simple/best_course_completion_time"] = lap_time[full_success_mask].min().item()

    failure_keys = {
        "wrong_gate": "failure/wrong_gate_rate",
        "collision": "failure/drone_collision_rate",
        "payload_collision": "failure/payload_collision_rate",
        "crashed_bounds": "failure/bounds_rate",
        "stalled": "failure/stall_rate",
    }
    for stat_key, log_key in failure_keys.items():
        stat = _flat_stat(stats, stat_key)
        if stat is not None:
            info[log_key] = (stat > 0.0).float().mean().item()

    payload_keys = {
        "payload_mean_swing_angle": "payload/mean_swing_angle",
        "payload_max_swing_angle": "payload/max_swing_angle",
        "payload_collision": "payload/contact_rate",
        "payload_miss": "payload/payload_miss_rate",
    }
    for stat_key, log_key in payload_keys.items():
        stat = _flat_stat(stats, stat_key)
        if stat is None:
            continue
        if stat_key in {"payload_collision", "payload_miss"}:
            info[log_key] = (stat > 0.0).float().mean().item()
        else:
            info[log_key] = stat.mean().item()

    return info


def launch_simulation_app(cfg):
    headless = bool(cfg.get("headless", True))
    config = {
        "headless": headless,
        "anti_aliasing": 0 if headless else 1,
        "disable_viewport_updates": headless,
        "multi_gpu": False,
        "active_gpu": 0,
        "physics_gpu": 0,
    }
    from isaacsim import SimulationApp

    return SimulationApp(config)


# def set_global_reproducibility(seed: int, deterministic: bool = True):
#     """Seed common RNGs and opt into deterministic torch kernels."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

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

    # set_global_reproducibility(cfg.seed, deterministic=cfg.get("deterministic", True))

    simulation_app = launch_simulation_app(cfg)

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
    
    # Only handle SIGTERM (SLURM preemption) — Isaac Sim sends SIGINT internally
    # during its own lifecycle so we must not intercept it.
    def signal_handler(sig, frame):
        logging.warning(f"Received SIGTERM. Saving config and exiting...")
        save_config_early()
        raise KeyboardInterrupt

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

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)
    if eval_interval <= 0 and cfg.task.get("eval_interval", -1) > 0:
        eval_interval = cfg.task.get("eval_interval")
    if save_interval <= 0 and cfg.task.get("save_interval", -1) > 0:
        save_interval = cfg.task.get("save_interval")

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

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE
    ):

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video
        video_array = render_callback.get_video_array(axes="t c h w")
        if video_array is not None:
            info["recording"] = wandb.Video(
                video_array,
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                format="mp4"
            )

        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        return info

    # Checkpoint tracking state
    # We keep at most 2 checkpoints on disk:
    #   1. "fastest speed that is >= 80% of best reward" checkpoint
    #   2. "latest" checkpoint
    # "best reward" is tracked purely to set the 80% threshold — that file is
    # not kept on disk unless it also happens to be the fastest or latest.
    ckpt_dir = "/tmp/106b_checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_reward = None        # highest mean return seen so far
    best_speed_ckpt = None    # (path, speed, reward) for the kept speed-champion
    latest_ckpt = None        # path of the most-recently saved checkpoint

    def _delete_if_exists(path):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def _save_checkpoint(frames, reward, speed):
        """Save a checkpoint and update tracking state. Returns the new path."""
        path = os.path.join(ckpt_dir, f"checkpoint_{frames}.pt")
        torch.save(policy.state_dict(), path)
        logging.info(f"Saved checkpoint to {path}")
        return path

    def _maybe_update_speed_ckpt(new_path, new_speed, new_reward, frames):
        nonlocal best_speed_ckpt, best_reward
        threshold = 0.8 * best_reward
        if new_reward < threshold:
            # Does not meet quality bar — only delete if it's not also latest
            if new_path != latest_ckpt:
                _delete_if_exists(new_path)
            return
        if best_speed_ckpt is None or new_speed > best_speed_ckpt[1]:
            old_path = best_speed_ckpt[0] if best_speed_ckpt else None
            best_speed_ckpt = (new_path, new_speed, new_reward)
            # Only delete the old speed ckpt if it's not also the latest
            if old_path and old_path != latest_ckpt:
                _delete_if_exists(old_path)
            logging.info(
                f"New fastest-quality checkpoint: speed={new_speed:.2f}, "
                f"reward={new_reward:.3f} (threshold={threshold:.3f}), frames={frames}"
            )
        else:
            # Not faster than current speed champion — only delete if it's not also latest
            if new_path != latest_ckpt:
                _delete_if_exists(new_path)

    loop_exception = None
    try:
        logging.info(
            f"Starting training loop: frames_per_batch={frames_per_batch}, "
            f"total_frames={total_frames}, num_envs={env.num_envs}"
        )
        pbar = tqdm(collector, total=total_frames//frames_per_batch)
        env.train()
        logging.info("Entering collector iteration.")
        for i, data in enumerate(pbar):
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            episode_stats.add(data.to_tensordict())

            current_reward = None
            current_speed = None
            if len(episode_stats) >= base_env.num_envs:
                episode_batch_stats = episode_stats.pop()
                stats = {
                    "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                    for k, v in episode_batch_stats.items(True, True)
                }
                info.update(stats)
                info.update(summarize_drone_race_stats(episode_batch_stats))

                # Auto-advance curriculum based on per-gate and full-course pass rates
                gate_pass_rates = {}
                for key, val in info.items():
                    if key.startswith("gates/gate") and key.endswith("_completion_rate"):
                        idx_str = key[len("gates/gate"):key.index("_completion_rate", len("gates/gate"))]
                        gate_pass_rates[int(idx_str)] = float(val)
                full_course_rate = info.get("simple/full_course_completion_rate", None)
                info.update(base_env.update_curriculum(gate_pass_rates, full_course_rate, env_frames=info.get("env_frames")))

                current_reward = info.get("train/stats.return", None)
                current_speed = info.get("simple/mean_speed", None)

            info.update(policy.train_op(data.to_tensordict()))

            if eval_interval > 0 and i % eval_interval == 0:
                logging.info(f"Eval at {collector._frames} steps.")
                # info.update(evaluate(seed=cfg.seed))
                info.update(evaluate())
                env.train()
                base_env.train()

            if save_interval > 0 and i % save_interval == 0:
                try:
                    reward = current_reward if current_reward is not None else 0.0
                    speed = current_speed if current_speed is not None else 0.0

                    # Update best reward
                    if best_reward is None or reward > best_reward:
                        best_reward = reward

                    new_path = _save_checkpoint(collector._frames, reward, speed)

                    # Update latest: delete old latest if it's not also the speed ckpt
                    old_latest = latest_ckpt
                    latest_ckpt = new_path
                    if old_latest and old_latest != (best_speed_ckpt[0] if best_speed_ckpt else None):
                        _delete_if_exists(old_latest)

                    # Check if this is also eligible to be the speed champion
                    if best_reward is not None and speed > 0:
                        _maybe_update_speed_ckpt(new_path, speed, reward, collector._frames)

                except AttributeError:
                    logging.warning(f"Policy {policy} does not implement `.state_dict()`")

            run.log(info)
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
            print(f"[train] epoch={i + 1} total_frames_processed={collector._frames}")

            pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

            if max_iters > 0 and i >= max_iters - 1:
                break

        logging.info("Collector iteration ended normally.")
        logging.info(f"Final Eval at {collector._frames} steps.")
        info = {"env_frames": collector._frames}
        # info.update(evaluate())
        # run.log(info)

        try:
            # Upload the two kept checkpoints as W&B artifacts
            kept_paths = set()
            if latest_ckpt and os.path.exists(latest_ckpt):
                kept_paths.add(latest_ckpt)
            if best_speed_ckpt and os.path.exists(best_speed_ckpt[0]):
                kept_paths.add(best_speed_ckpt[0])

            for ckpt_path in kept_paths:
                label = "latest" if ckpt_path == latest_ckpt else "fastest_quality"
                model_artifact = wandb.Artifact(
                    f"{cfg.task.name}-{cfg.algo.name.lower()}-{label}",
                    type="model",
                    description=f"{cfg.task.name}-{cfg.algo.name.lower()} ({label})",
                    metadata=dict(cfg),
                )
                model_artifact.add_file(ckpt_path)
                run.log_artifact(model_artifact)
                logging.info(f"Uploaded {label} checkpoint: {ckpt_path}")
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
