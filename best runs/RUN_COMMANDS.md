# Run Commands

Keep `checkpoint_929562624.trainer.pt` in this folder next to `checkpoint_929562624.pt`.
Copy only the command line itself. Do not paste a separate leading `bash` line.
Visible camera modes in this repo: `track` (default, non-follow), `follow`, `side`, `fixed`.
The visible-window commands below keep the original real-time renderer but force sharper `RTXAA` rendering at `2560x1440`.
These are the current verified commands.
The second command keeps the original `track` view.

## With follow camera

```
cd "/home/cc/ee106b/sp26/class/ee106b-abg/106bFinalProject/scripts" && PYTHONPATH="/home/cc/ee106b/sp26/class/ee106b-abg/106bFinalProject" python3 -u play.py task=DroneRace headless=false task.env.num_envs=1 play_exploration=RANDOM total_frames=2048 +play_start_gate=1 +play_camera_mode=follow +renderer=RaytracedLighting +anti_aliasing=4 +width=2560 +height=1440 +window_width=2560 +window_height=1440 "viewer.resolution=[2560,1440]" "algo.checkpoint_path=/home/cc/ee106b/sp26/class/ee106b-abg/106bFinalProject/best runs/checkpoint_929562624.pt" wandb.mode=disabled
```

## With visible window, non-follow track camera

```
cd "/home/cc/ee106b/sp26/class/ee106b-abg/106bFinalProject/scripts" && PYTHONPATH="/home/cc/ee106b/sp26/class/ee106b-abg/106bFinalProject" python3 -u play.py task=DroneRace headless=false task.env.num_envs=1 play_exploration=RANDOM total_frames=2048 +play_start_gate=1 +play_camera_mode=track +renderer=RaytracedLighting +anti_aliasing=4 +width=2560 +height=1440 +window_width=2560 +window_height=1440 "viewer.resolution=[2560,1440]" "algo.checkpoint_path=/home/cc/ee106b/sp26/class/ee106b-abg/106bFinalProject/best runs/checkpoint_929562624.pt" wandb.mode=disabled
```
