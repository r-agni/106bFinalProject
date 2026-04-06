#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/launch_hover_racetrack.sh [--smoke] [--dry-run] [--hover-num-envs N] [--race-num-envs N]

Options:
  --smoke              Short validation run (max_iters=3, save_interval=2).
  --dry-run            Print commands without launching.
  --hover-num-envs N   Override Hover num envs (default: 128).
  --race-num-envs N    Override DroneRace num envs (default: 128).
  -h, --help           Show this help.
USAGE
}

SMOKE=false
DRY_RUN=false
HOVER_NUM_ENVS=128
RACE_NUM_ENVS=128
GPU_ID=0
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --hover-num-envs)
      HOVER_NUM_ENVS="$2"
      shift 2
      ;;
    --race-num-envs)
      RACE_NUM_ENVS="$2"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_OMNI_ROOT="/workspace/omni_drones"

if [[ -n "${OMNI_ROOT:-}" ]]; then
  OMNI_ROOT="$OMNI_ROOT"
elif [[ -d "$DEFAULT_OMNI_ROOT/scripts" && -f "$DEFAULT_OMNI_ROOT/.env" ]]; then
  OMNI_ROOT="$DEFAULT_OMNI_ROOT"
else
  OMNI_ROOT="$REPO_ROOT"
fi

TRAIN_DIR="$OMNI_ROOT/scripts"
LOG_DIR="$OMNI_ROOT/.logs"
ENV_FILE="$OMNI_ROOT/.env"

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "ERROR: Training directory not found: $TRAIN_DIR" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Expected env file at $ENV_FILE but it was not found." >&2
  exit 1
fi

ISAAC_SETUP_SCRIPT="${ISAACSIM_PATH:-/workspace/isaacsim}/setup_conda_env.sh"
if [[ -f "$ISAAC_SETUP_SCRIPT" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ISAAC_SETUP_SCRIPT"
  set -u
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is missing after sourcing $ENV_FILE" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$OMNI_ROOT/.hydra_outputs"

# Ensure only one launcher instance is active at a time.
LOCK_DIR="$LOG_DIR/launch_hover_racetrack.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: Another launch_hover_racetrack.sh instance is already running." >&2
  exit 1
fi
cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

export CUDA_VISIBLE_DEVICES="$GPU_ID"
cache_dir="${XDG_CACHE_HOME:-$OMNI_ROOT/.cache}"
if ! mkdir -p "$cache_dir" 2>/dev/null; then
  cache_dir="/tmp/od_cache"
  mkdir -p "$cache_dir"
fi
export XDG_CACHE_HOME="$cache_dir"

if [[ -d "/run/host/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="/run/host/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "/run/host/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="/run/host/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: Could not find '$PYTHON_BIN' or 'python3' in PATH." >&2
    exit 1
  fi
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
GROUP_NAME="hover-dronerace-${TIMESTAMP}"

COMMON_ARGS=(
  "algo=ppo"
  "headless=true"
  "seed=0"
  "wandb.mode=online"
  "wandb.group=${GROUP_NAME}"
)

if [[ "$SMOKE" == "true" ]]; then
  COMMON_ARGS+=("max_iters=3" "save_interval=2")
fi

launch_one() {
  local key="$1"
  local run_id
  local -a cmd

  case "$key" in
    hover)
      run_id="Hover-ppo-${TIMESTAMP}"
      cmd=(
        "$PYTHON_BIN" train.py
        "task=Hover"
        "task.env.num_envs=${HOVER_NUM_ENVS}"
        "wandb.run_name=${run_id}"
        "hydra.run.dir=${OMNI_ROOT}/.hydra_outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_${run_id}"
        "${COMMON_ARGS[@]}"
      )
      ;;
    dronerace)
      run_id="DroneRace-ppo-${TIMESTAMP}"
      cmd=(
        "$PYTHON_BIN" train.py
        "task=DroneRace"
        "task.env.num_envs=${RACE_NUM_ENVS}"
        "wandb.run_name=${run_id}"
        "hydra.run.dir=${OMNI_ROOT}/.hydra_outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_${run_id}"
        "${COMMON_ARGS[@]}"
      )
      ;;
    *)
      echo "ERROR: Unknown key $key" >&2
      return 1
      ;;
  esac

  local log_file="$LOG_DIR/${run_id}.log"

  if [[ "$DRY_RUN" == "true" ]]; then
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "          log -> $log_file"
    return 0
  fi

  echo "[launch] $run_id"
  echo "         log -> $log_file"
  if ! (
    cd "$TRAIN_DIR"
    "${cmd[@]}"
  ) > "$log_file" 2>&1; then
    echo "[failed] $run_id"
    return 1
  fi
  echo "[done] $run_id"
}

echo "OMNI_ROOT: $OMNI_ROOT"
echo "Training dir: $TRAIN_DIR"
echo "W&B group: $GROUP_NAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "XDG_CACHE_HOME: $XDG_CACHE_HOME"
echo "Hover num_envs: $HOVER_NUM_ENVS"
echo "DroneRace num_envs: $RACE_NUM_ENVS"
echo "Mode: sequential (Hover -> DroneRace)"

failures=0
if ! launch_one hover; then
  failures=$((failures + 1))
fi
if (( failures == 0 )); then
  if ! launch_one dronerace; then
    failures=$((failures + 1))
  fi
fi

if [[ "$DRY_RUN" == "false" && $failures -eq 0 ]]; then
  echo "All runs finished successfully."
elif [[ "$DRY_RUN" == "false" ]]; then
  echo "Completed with $failures failed run(s)." >&2
  exit 1
fi
