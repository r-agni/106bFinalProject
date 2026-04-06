#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/launch_parallel_dronerace.sh [--smoke] [--dry-run] [--num-envs N]

Options:
  --smoke       Launch one short smoke run (max_iters=3, save_interval=2).
  --dry-run     Print commands without executing them.
  --num-envs N  Override task.env.num_envs (default: 128).
  -h, --help    Show this help message.
USAGE
}

SMOKE=false
DRY_RUN=false
NUM_ENVS=128
MAX_PARALLEL=2
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
HYDRA_BASE_DIR="$OMNI_ROOT/.hydra_outputs"
ENV_FILE="$OMNI_ROOT/.env"

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "ERROR: Training directory not found: $TRAIN_DIR" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Expected env file at $ENV_FILE but it was not found." >&2
  exit 1
fi

# Source Isaac Sim environment when available so `isaacsim` is importable in
# non-interactive shells (e.g., podman exec / distrobox exec).
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

mkdir -p "$LOG_DIR" "$HYDRA_BASE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
cache_dir="${XDG_CACHE_HOME:-$OMNI_ROOT/.cache}"
if ! mkdir -p "$cache_dir" 2>/dev/null; then
  cache_dir="/tmp/od_cache"
  mkdir -p "$cache_dir"
fi
export XDG_CACHE_HOME="$cache_dir"

# In some distrobox setups, NVIDIA user-space libs are exposed via /run/host.
# Prepend these paths when present so Warp/Isaac Sim can load libcuda.so.
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
GROUP_NAME="dronerace-ppo-hparam-${TIMESTAMP}"

BASE_ARGS=(
  "task=DroneRace"
  "algo=ppo"
  "headless=true"
  "task.env.num_envs=${NUM_ENVS}"
  "seed=0"
  "wandb.mode=online"
  "wandb.group=${GROUP_NAME}"
)

if [[ "$SMOKE" == "true" ]]; then
  BASE_ARGS+=("max_iters=3" "save_interval=2")
fi

RUN_KEYS=(
  "A_lr3e-4_ent3e-3"
  "B_lr3e-4_ent1e-3"
  "C_lr1e-4_ent3e-3"
  "D_lr1e-4_ent1e-3"
)

if [[ "$SMOKE" == "true" ]]; then
  RUN_KEYS=("A_lr3e-4_ent3e-3")
fi

run_overrides() {
  local key="$1"
  case "$key" in
    A_lr3e-4_ent3e-3)
      echo "algo.actor.lr=0.0003 algo.critic.lr=0.0003 algo.entropy_coef=0.003"
      ;;
    B_lr3e-4_ent1e-3)
      echo "algo.actor.lr=0.0003 algo.critic.lr=0.0003 algo.entropy_coef=0.001"
      ;;
    C_lr1e-4_ent3e-3)
      echo "algo.actor.lr=0.0001 algo.critic.lr=0.0001 algo.entropy_coef=0.003"
      ;;
    D_lr1e-4_ent1e-3)
      echo "algo.actor.lr=0.0001 algo.critic.lr=0.0001 algo.entropy_coef=0.001"
      ;;
    *)
      echo "ERROR: unknown run key: $key" >&2
      return 1
      ;;
  esac
}

launch_one() {
  local key="$1"
  local run_id="${key}-${TIMESTAMP}"
  local log_file="$LOG_DIR/${run_id}.log"
  local hydra_run_dir="hydra.run.dir=${OMNI_ROOT}/.hydra_outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_${run_id}"
  local override_string
  override_string="$(run_overrides "$key")"
  local -a override_args
  read -r -a override_args <<< "$override_string"

  local -a cmd=(
    "$PYTHON_BIN" train.py
    "${BASE_ARGS[@]}"
    "wandb.run_name=${run_id}"
    "$hydra_run_dir"
    "${override_args[@]}"
  )

  if [[ "$DRY_RUN" == "true" ]]; then
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "          log -> $log_file"
    return 0
  fi

  (
    cd "$TRAIN_DIR"
    "${cmd[@]}"
  ) > "$log_file" 2>&1 &

  local pid=$!
  echo "[launch] $run_id (pid=$pid)"
  echo "         log -> $log_file"
}

echo "OMNI_ROOT: $OMNI_ROOT"
echo "Training dir: $TRAIN_DIR"
echo "W&B group: $GROUP_NAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo "task.env.num_envs: $NUM_ENVS"
echo "Python executable: $PYTHON_BIN"
echo "XDG_CACHE_HOME: $XDG_CACHE_HOME"

failures=0

for key in "${RUN_KEYS[@]}"; do
  launch_one "$key"

  if [[ "$DRY_RUN" == "true" ]]; then
    continue
  fi

  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
  done
done

if [[ "$DRY_RUN" == "false" ]]; then
  while (( $(jobs -pr | wc -l) > 0 )); do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
  done

  if (( failures > 0 )); then
    echo "Completed with $failures failed run(s)." >&2
    echo "If failures were OOM-related, rerun with: --num-envs 96" >&2
    exit 1
  fi

  echo "All runs finished successfully."
fi
