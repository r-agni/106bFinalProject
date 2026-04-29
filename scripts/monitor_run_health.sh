#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_DIR [INTERVAL_SEC] [STALE_AFTER_SEC]" >&2
  exit 2
fi

run_dir="$1"
interval_sec="${2:-300}"
stale_after_sec="${3:-420}"
files_dir="$run_dir/files"
output_log="$files_dir/output.log"
monitor_log="$run_dir/monitor.log"
status_file="$run_dir/monitor.status"
pid_file="$run_dir/monitor.pid"

mkdir -p "$run_dir"

if [[ -f "$pid_file" ]]; then
  existing_pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "monitor already running with pid $existing_pid" >&2
    exit 1
  fi
fi

echo "$$" > "$pid_file"
cleanup() {
  rm -f "$pid_file"
}
trap cleanup EXIT

latest_checkpoint() {
  find "$files_dir" -maxdepth 1 -type f -name 'checkpoint_*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

latest_epoch_line() {
  if [[ -f "$output_log" ]]; then
    tail -n 400 "$output_log" | rg '\[train\] epoch=' | tail -n 1 || true
  fi
}

format_status() {
  local now_iso="$1"
  local state="$2"
  local ckpt_name="$3"
  local ckpt_age="$4"
  local log_age="$5"
  local epoch_line="$6"
  printf '%s state=%s checkpoint=%s ckpt_age_sec=%s log_age_sec=%s %s\n' \
    "$now_iso" "$state" "$ckpt_name" "$ckpt_age" "$log_age" "$epoch_line"
}

while true; do
  now_epoch="$(date +%s)"
  now_iso="$(date --iso-8601=seconds)"
  ckpt_path="$(latest_checkpoint)"
  ckpt_name="none"
  ckpt_age="na"
  log_age="na"
  state="missing"
  epoch_line="$(latest_epoch_line)"

  if [[ -n "$ckpt_path" && -f "$ckpt_path" ]]; then
    ckpt_name="$(basename "$ckpt_path")"
    ckpt_mtime="$(stat -c %Y "$ckpt_path")"
    ckpt_age="$((now_epoch - ckpt_mtime))"
    state="ok"
  fi

  if [[ -f "$output_log" ]]; then
    log_mtime="$(stat -c %Y "$output_log")"
    log_age="$((now_epoch - log_mtime))"
    if [[ "$state" == "missing" ]]; then
      state="log_only"
    fi
  fi

  if [[ "$ckpt_age" != "na" && "$ckpt_age" -gt "$stale_after_sec" ]] \
    && [[ "$log_age" != "na" && "$log_age" -gt "$stale_after_sec" ]]; then
    state="stalled"
  fi

  status_line="$(format_status "$now_iso" "$state" "$ckpt_name" "$ckpt_age" "$log_age" "$epoch_line")"
  printf '%s\n' "$status_line" | tee -a "$monitor_log" > "$status_file"

  sleep "$interval_sec"
done
