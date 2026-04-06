#!/usr/bin/env bash
set -euo pipefail

SLEEP_SEC=30
MAX_RETRIES=2
OMNI_ROOT="${OMNI_ROOT:-/workspace/omni_drones}"
LAUNCHER_LOG_REL=".logs/launcher-hover-race-current.log"
LOCK_DIR_REL=".logs/launch_hover_racetrack.lock"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/monitor_hover_race.sh [--sleep-sec N] [--max-retries N]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sleep-sec)
      SLEEP_SEC="$2"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$OMNI_ROOT"
mkdir -p .logs

attempt=0
while true; do
  ts="$(date +"%Y-%m-%d %H:%M:%S")"

  if pgrep -f "train.py task=Hover" >/dev/null || \
     pgrep -f "train.py task=DroneRace" >/dev/null || \
     pgrep -f "Hover-ppo-" >/dev/null || \
     pgrep -f "DroneRace-ppo-" >/dev/null || \
     pgrep -f "scripts/launch_hover_racetrack.sh" >/dev/null; then
    echo "[$ts] monitor: training/launcher active"
  else
    if grep -q "All runs finished successfully." .logs/launcher-hover-race*.log 2>/dev/null; then
      echo "[$ts] monitor: completed successfully"
      exit 0
    fi

    attempt=$((attempt + 1))
    if (( attempt > MAX_RETRIES )); then
      echo "[$ts] monitor: retries exhausted, giving up"
      exit 1
    fi

    if (( attempt == 1 )); then
      HN=96
      RN=96
    else
      HN=64
      RN=64
    fi

    if [[ -d "$LOCK_DIR_REL" ]]; then
      echo "[$ts] monitor: removing stale launcher lock at $LOCK_DIR_REL"
      rm -rf "$LOCK_DIR_REL"
    fi

    retry_log=".logs/launcher-hover-race-retry${attempt}.log"
    echo "[$ts] monitor: detected failure, relaunching with --hover-num-envs ${HN} --race-num-envs ${RN}"
    nohup bash scripts/launch_hover_racetrack.sh --hover-num-envs "$HN" --race-num-envs "$RN" > "$retry_log" 2>&1 &
  fi

  sleep "$SLEEP_SEC"
done
