# Hover policy in Isaac Sim with viewport (for OBS / Xbox Game Bar recording).
# Run:  powershell -ExecutionPolicy Bypass -File .\scripts\run_hover_playback.ps1
# From repo root, or:  cd scripts; .\run_hover_playback.ps1

$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "scripts")

$ckpt = Join-Path $Root "checkpoint_final.pt"
$outMp4 = Join-Path $Root "hover_playback.mp4"
python play.py `
  record_video=true `
  headless=false `
  task.env.num_envs=1 `
  total_frames=8000 `
  "algo.checkpoint_path=$ckpt" `
  wandb.mode=disabled `
  "video_path=$outMp4"
