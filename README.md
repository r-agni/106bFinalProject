# OmniDrones on your PC: Isaac Sim, playback, and recording

This guide is for running this repo **on your own Windows machine** with a **local NVIDIA GPU**, using **Isaac Sim** (pip install) and saving **MP4** recordings of the trained hover policy.

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

Put your trained policy next to the repo or pass an absolute path:

- Example: `c:/Users/agni_/Documents/106bFinalProject/checkpoint_final.pt`  
- This must be a **PPO `state_dict`** saved the same way as `scripts/train.py` (e.g. `checkpoint_final.pt`).

Playback uses the **Hover** task and **ppo** config unless you override Hydra (`task=...`, `algo=...`). If you trained with extra overrides, use the **same** overrides for `play.py`.

---

## 4. Run policy in Isaac Sim (no video file)

`play.py` uses Hydra with [`scripts/train.yaml`](scripts/train.yaml). Defaults include **`headless: true`** and a large `total_frames` from the task YAML—override them for a short demo.

```bash
cd c:/Users/agni_/Documents/106bFinalProject/scripts
python play.py headless=true task.env.num_envs=1 total_frames=2048 ^
  algo.checkpoint_path=c:/Users/agni_/Documents/106bFinalProject/checkpoint_final.pt ^
  wandb.mode=disabled
```

(Bash: use `\` line continuations instead of `^`.)

- **`task.env.num_envs=1`**: one drone, easier to watch or record.
- **`play_exploration`**: defaults to **`MODE`** (mean / mode action—stable hover). Use `play_exploration=RANDOM` only if you want deliberate noise.

---

## 5. Record an MP4 on disk

Recording needs a **viewport** and **Omniverse Replicator** RGB. With **`record_video=true`**, `play.py` sets **`sim.enable_replicator=True`** and forces **`headless=false`**.

```bash
cd c:/Users/agni_/Documents/106bFinalProject/scripts
python play.py record_video=true task.env.num_envs=1 total_frames=2048 ^
  record_video_max_steps=800 video_frame_interval=2 ^
  algo.checkpoint_path=c:/Users/agni_/Documents/106bFinalProject/checkpoint_final.pt ^
  wandb.mode=disabled ^
  video_path=c:/Users/agni_/Documents/106bFinalProject/hover_playback.mp4
```

| Override | Meaning |
|----------|---------|
| `video_path` | Output file (use an **absolute** path if you do not want it under Hydra’s run directory). |
| `record_video_max_steps` | Length of the **extra** rollout used only for frames (after the main collector loop). |
| `video_frame_interval` | Save one frame every **N** env steps during that rollout. |
| `video_fps` | Encoder FPS (default **30** in [`scripts/train.yaml`](scripts/train.yaml)). |

Defaults for these live in [`scripts/train.yaml`](scripts/train.yaml) (`record_video`, `video_path`, `play_exploration`, etc.).

**Where is the video?** Exactly where you set `video_path`—e.g. repo root `hover_playback.mp4`. If MP4 encoding fails, the code may fall back to a **GIF** with the same base name.

**Time:** First launch loads Kit and extensions; a full run can take **many minutes** (simulation + encoding).

### Helper scripts (same idea, fixed paths)

- [`scripts/run_hover_playback.bat`](scripts/run_hover_playback.bat)  
- [`scripts/run_hover_playback.ps1`](scripts/run_hover_playback.ps1)  

They assume `checkpoint_final.pt` at the **repo root** and write **`hover_playback.mp4`** there.

---

## 6. Hydra output directory

[`scripts/train.yaml`](scripts/train.yaml) sets:

```yaml
hydra.run.dir: .hydra_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

Logs and Hydra artifacts go under **`scripts/.hydra_outputs/...`** when you run from `scripts/`. This avoids Linux-only `/workspace/...` paths.

---

## 7. What you see in the Hover scene

- **Two drone-shaped meshes:** The second is a **static target marker** at the goal pose (same USD as the robot, **no collision**, **no gravity**). Only one is the controlled agent. See [`omni_drones/envs/single/hover.py`](omni_drones/envs/single/hover.py) (`target_vis_prim`).
- **Aggressive motion after resets:** Each episode the drone is **respawned randomly** in a large box (`init_pos_dist` in the same file), so it may **move hard** to rejoin the target—that is training-style randomization, not a “hold still forever” spawn.

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

