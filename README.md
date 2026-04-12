# OmniDrones on your PC: Drone Racing RL Training and Playback

This guide covers **training** a drone racing policy from scratch and **playing back** a trained checkpoint, on a Windows machine with a local NVIDIA GPU.

---

## Training the Drone Racing Policy

### Quick start

```bash
cd c:/Users/agni_/Documents/106bFinalProject/scripts
python train.py task=DroneRace wandb.mode=disabled
```

This uses all defaults from `cfg/task/DroneRace.yaml` and `cfg/algo/DroneRace.yaml`:
- **500 parallel environments**, 100 Hz policy rate
- **400 million frames** total (`total_frames: 400_000_000`)
- Checkpoints saved every 100 rollouts (~6.4 M frames) under the W&B run directory
- Expected wall-clock time: **~6–8 hours** on a modern GPU (RTX 3090/4090 class)

> **Why 400 M frames?** A pilot run at 200 M frames showed the policy still improving with no lap completions. The drone needs roughly:
> - **0–10 M frames** to learn basic gate-crossing (curriculum phase)
> - **10–50 M frames** to chain gates reliably
> - **50–150 M frames** to push speed and reduce crashes
> - **150–400 M frames** to optimise lap times and achieve consistent completions
>
> Running fewer than ~300 M frames will likely produce a drone that navigates toward gates but rarely completes a full lap.

### With W&B logging (recommended)

Set your W&B credentials in `.env` at the repo root:

```bash
WANDB_API_KEY=<your_key>
WANDB_PROJECT=droneRacing
```

Then train:

```bash
cd c:/Users/agni_/Documents/106bFinalProject/scripts
python train.py task=DroneRace
```

Key metrics to watch during training:
| Metric | Healthy sign |
|--------|-------------|
| `reward/gates_cumul` | Must be > 0 by 5 M frames |
| `crash/distance_rate` | Should not spike above 50% |
| `race/gates_passed_per_ep` | > 2.0 by 50 M frames |
| `race/mean_speed_ms` | > 4.0 m/s by 100 M frames |
| `race/lap_completion_rate` | > 0 by 150 M frames |

If `reward/gates_cumul` is still 0 at 5 M frames, something is wrong with gate detection — stop and debug before continuing.

### Resume a stopped run

```bash
cd c:/Users/agni_/Documents/106bFinalProject/scripts
python train.py task=DroneRace \
  algo.checkpoint_path=c:/Users/agni_/Documents/106bFinalProject/wandb/run-<id>/files/checkpoint_<frames>.pt
```

### Faster experiment (reduced environments, fewer frames)

For a quick sanity check that training is working (not expected to produce a racing policy):

```bash
cd c:/Users/agni_/Documents/106bFinalProject/scripts
python train.py task=DroneRace \
  task.env.num_envs=100 \
  total_frames=50_000_000 \
  wandb.mode=disabled
```

---

## 1. Prerequisites

- **Windows** with an **NVIDIA GPU** and current **Game Ready / Studio** drivers.
- **Python 3.12** (this project was verified with **Anaconda** base).
- **Isaac Sim** available to that same Python, e.g. NVIDIA’s **pip** workflow so that:

  ```bash
  python -c "from isaacsim import SimulationApp; print('ok')"
  ```

  runs without errors (first import can take ~20s).

If `import isaacsim` fails, install or repair Isaac Sim using [NVIDIA Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/) for your version before continuing.

### Local Isaac Sim path (pip install on this machine)

With **Isaac Sim 6.x installed via pip** into **Anaconda**, the Python package and Kit runtime live under **`site-packages`**, not under Omniverse Launcher’s `%LOCALAPPDATA%\ov\pkg\...` tree.

| What | Path on this PC |
|------|-----------------|
| **`isaacsim` Python package** | `C:\Users\agni_\anaconda3\Lib\site-packages\isaacsim` |
| **Kit / binaries / exts** (typical) | `C:\Users\agni_\anaconda3\Lib\site-packages\isaacsim\kit` |

To **print the folder for whatever `python` you use**:

```bash
python -c "import isaacsim, os; print(os.path.dirname(isaacsim.__file__))"
```

To see pip metadata (version, declared `Location`):

```bash
pip show isaacsim
```

If you use another conda env or venv, run those commands **after** `conda activate ...` so the path matches that environment.

---

## 2. Install this package (editable)

From the **repository root** (folder that contains `setup.py` and `omni_drones/`):

```bash
cd c:/Users/agni_/Documents/106bFinalProject
pip install -e .
```

That pulls dependencies from `setup.py` (Hydra, TorchRL, wandb, imageio, etc.).

### Broken editable install (`egg-link` pointing at WSL or old path)

If `pip install -e .` fails with a missing path (often a `\\wsl.localhost\...` or old clone location), remove the stale link in your environment’s `site-packages`:

- Delete **`omni-drones.egg-link`** (name may vary) under  
  `C:\Users\<you>\anaconda3\Lib\site-packages\`  
  then run `pip install -e .` again from this repo.

---

## 3. Checkpoint

After training, your checkpoint will be at:

```
wandb/run-<date>_<time>-<run_id>/files/checkpoint_final.pt
```

or at intermediate saves:

```
wandb/run-<date>_<time>-<run_id>/files/checkpoint_<frames>.pt
```

A well-trained checkpoint requires at least **~300 M frames** of `DroneRace` training before you can expect to see lap completions. The final checkpoint from a 400 M frame run is recommended.

---

## 4. Run policy in Isaac Sim (no video file)

`play.py` uses Hydra with [`scripts/train.yaml`](scripts/train.yaml). Use `task=DroneRace` to match the training task.

```bash
cd c:/Users/agni_/Documents/106bFinalProject/scripts
python play.py task=DroneRace headless=false task.env.num_envs=1 \
  algo.checkpoint_path=c:/Users/agni_/Documents/106bFinalProject/wandb/run-<id>/files/checkpoint_final.pt \
  wandb.mode=disabled
```

- **`task.env.num_envs=1`**: single drone — easier to observe the racing behaviour.
- **`headless=false`**: opens the Isaac Sim viewport so you can watch the drone fly.
- **`play_exploration=MODE`** (default): uses the deterministic mean action — the cleanest racing behaviour.

---

## 5. Record an MP4 of a race

Recording needs a **viewport** and **Omniverse Replicator** RGB. With **`record_video=true`**, `play.py` forces **`headless=false`**.

```bash
cd c:/Users/agni_/Documents/106bFinalProject/scripts
python play.py task=DroneRace record_video=true task.env.num_envs=1 \
  record_video_max_steps=3000 video_frame_interval=2 \
  algo.checkpoint_path=c:/Users/agni_/Documents/106bFinalProject/wandb/run-<id>/files/checkpoint_final.pt \
  wandb.mode=disabled \
  video_path=c:/Users/agni_/Documents/106bFinalProject/race_playback.mp4
```

`record_video_max_steps=3000` captures about 30 seconds at 100 Hz — enough to show several gate crossings and at least one full lap from a well-trained policy.

| Override | Meaning |
|----------|---------|
| `video_path` | Output file — use an **absolute** path. |
| `record_video_max_steps` | Steps in the recording rollout. 3000 ≈ 30 s at 100 Hz. |
| `video_frame_interval` | Save one frame every **N** env steps (2 = 50 fps output). |
| `video_fps` | Encoder FPS (default **30** in [`scripts/train.yaml`](scripts/train.yaml)). |

**Where is the video?** Exactly where you set `video_path`. If MP4 encoding fails the code may fall back to a **GIF** with the same base name.

**Time:** First launch loads Kit and extensions — allow several minutes before the drone appears.

---

## 6. Hydra output directory

[`scripts/train.yaml`](scripts/train.yaml) sets:

```yaml
hydra.run.dir: .hydra_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

Logs and Hydra artifacts go under **`scripts/.hydra_outputs/...`** when you run from `scripts/`. This avoids Linux-only `/workspace/...` paths.

---

## 7. What you see in the DroneRace scene

- **One Iris quadrotor** flying a four-gate diamond circuit. Gates are 2.5 m × 2.5 m at 1 m altitude, spaced 7.07 m apart with 90° turn angles between them.
- **Episode resets:** The drone spawns 1.5 m behind gate 0 (during bootstrap) or behind a random gate (after 10 M frames). After each episode end the drone snaps back to its start position — this is normal.
- **A well-trained policy** (300 M+ frames) should visibly bank through corners, maintain ~4–6 m/s, and complete full laps. An undertrained policy may hover near a gate or crash frequently.

---

## 8. Optional: screen recording instead of Replicator

If `record_video` or Replicator causes issues, run with **`headless=false`** and **`record_video=false`**, then capture the Isaac Sim window with **OBS Studio** or **Win+G (Game Bar)**.

---

## 9. Repo changes relevant to Isaac Sim 6 (pip)

This codebase was adjusted for **Isaac Sim 6.x** style APIs, for example:

- Removed deprecated **`isaacsim.core.utils.nucleus`** usage in [`omni_drones/utils/kit.py`](omni_drones/utils/kit.py).
- **`ArticulationView.initialize`** delegates to the upstream Isaac implementation using **`SimulationManager`** ([`omni_drones/views/__init__.py`](omni_drones/views/__init__.py)).
- **`IsaacEnv.render`** accepts Replicator **`get_data()`** returning a **NumPy array** ([`omni_drones/envs/isaac_env.py`](omni_drones/envs/isaac_env.py)).

If you use an older Isaac layout, you may need a branch or version matched to your install.

