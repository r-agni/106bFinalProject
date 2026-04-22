#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/launch_dronerace_figure8.sh [--smoke] [--dry-run] [--num-envs N]

Options:
  --smoke       Launch a short validation run (max_iters=3, save_interval=2).
  --dry-run     Print the training command without launching it.
  --num-envs N  Override task.env.num_envs (default: 512).
  -h, --help    Show this help message.
USAGE
}

SMOKE=false
DRY_RUN=false
NUM_ENVS=512
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
    --num-envs)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --num-envs requires a value." >&2
        exit 1
      fi
      NUM_ENVS="$2"
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
RUN_ID="DroneRaceFigure8-ppo-${TIMESTAMP}"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"

CMD=(
  "$PYTHON_BIN" train.py
  "task=DroneRaceFigure8"
  "algo=ppo"
  "headless=true"
  "task.env.num_envs=${NUM_ENVS}"
  "seed=0"
  "wandb.mode=online"
  "wandb.group=dronerace-figure8-${TIMESTAMP}"
  "wandb.run_name=${RUN_ID}"
  "hydra.run.dir=${OMNI_ROOT}/.hydra_outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_${RUN_ID}"
)

if [[ "$SMOKE" == "true" ]]; then
  CMD+=("max_iters=3" "save_interval=2")
fi

echo "OMNI_ROOT: $OMNI_ROOT"
echo "Training dir: $TRAIN_DIR"
echo "Run id: $RUN_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "XDG_CACHE_HOME: $XDG_CACHE_HOME"
echo "task.env.num_envs: $NUM_ENVS"

if [[ "$DRY_RUN" == "true" ]]; then
  printf '[dry-run] '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  echo "          log -> $LOG_FILE"
  exit 0
fi

echo "[launch] $RUN_ID"
echo "         log -> $LOG_FILE"
(
  cd "$TRAIN_DIR"
  "${CMD[@]}"
) > "$LOG_FILE" 2>&1
echo "[done] $RUN_ID"
