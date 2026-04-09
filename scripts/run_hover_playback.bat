@echo off
REM Hover policy in Isaac Sim with viewport (for OBS / Game Bar recording).
REM Usage: run from anywhere, or double-click from Explorer.

set "ROOT=%~dp0.."
cd /d "%ROOT%\scripts"

python play.py record_video=true headless=false task.env.num_envs=1 total_frames=8000 ^
  algo.checkpoint_path=%ROOT%\checkpoint_final.pt ^
  wandb.mode=disabled ^
  video_path=%ROOT%\hover_playback.mp4

pause
