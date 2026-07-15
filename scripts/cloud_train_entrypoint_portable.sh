#!/usr/bin/env bash
# Portable TRON2 training entrypoint for container and cloud platforms.

set -Eeuo pipefail

PROJECT_DIR="${OPENPI_PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${HF_LEROBOT_HOME:-/data/input}}"
WEIGHT_PATH="${WEIGHT_PATH:-/data/checkpoint/params}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/output}"

REPO_ID=""
EXP_NAME=""
PROMPT="Perform the configured manipulation task"
STEPS=20000
SAVE_INTERVAL=5000
BATCH_SIZE=32
ACTION_HORIZON=50
USE_DELTA=false
RTC_DELAY=""
PROMPT_FROM_TASK=false
MAX_FRAMES=""
SKIP_NORM=false
NAME_OVERRIDE=""
TASK_CONFIG=""
RESUME=false
OVERWRITE=false
USE_WANDB=false
DRY_RUN=false
GENERATED_TASK_CONFIG=false

usage() {
  cat <<'EOF'
Usage:
  scripts/cloud_train_entrypoint_portable.sh --repo-id ID --exp NAME [options]
  scripts/cloud_train_entrypoint_portable.sh --task-config FILE [--exp NAME] [options]

Default platform paths:
  dataset root:  --data-dir, DATA_DIR, or HF_LEROBOT_HOME; fallback /data/input
  model params:  /data/checkpoint/params
  output root:   /data/output

Options:
  --repo-id ID          LeRobot dataset ID below the dataset root
  --exp NAME            Experiment and checkpoint directory name
  --prompt TEXT         Fallback task prompt
  --prompt-from-task    Use each LeRobot episode's task text as the prompt
  --steps N             Training steps (default: 20000)
  --save-interval N     Checkpoint interval (default: 5000)
  --batch-size N        Global batch size (default: 32)
  --action-horizon N    Action horizon (default: 50)
  --rtc-delay N         Enable training-time RTC with this simulated delay
  --delta               Train with delta joint actions
  --max-frames N        Maximum frames used for normalization statistics
  --skip-norm           Skip normalization-statistics computation
  --name NAME           Override the generated training config name
  --data-dir DIR        Override the LeRobot dataset root and HF_LEROBOT_HOME
  --weight PATH         Override the initialization params path
  --output-dir DIR      Override the output root
  --task-config FILE    Use an existing task YAML instead of generating one
  --resume              Resume an existing checkpoint; never overwrites it
  --overwrite           Replace an existing experiment checkpoint
  --wandb               Enable Weights & Biases (disabled by default)
  --dry-run             Validate inputs and print commands without training
  -h, --help            Show this help

Example matching a platform that stores LeRobot datasets under /path/to/datasets:
  scripts/cloud_train_entrypoint_portable.sh \
    --data-dir /path/to/datasets --repo-id my_dataset --exp example \
    --weight /models/openpi-assets/checkpoints/pi_base/params \
    --steps 30000 --rtc-delay 10 --action-horizon 30 \
    --batch-size 64 --prompt-from-task --max-frames 100000
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [ -n "$value" ] || die "$option requires a value"
}

require_positive_integer() {
  local option="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$option must be a positive integer: $value"
}

require_nonnegative_integer() {
  local option="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$option must be a non-negative integer: $value"
}

print_command() {
  local label="$1"
  shift
  printf '%s' "$label"
  printf ' %q' "$@"
  printf '\n'
}

cleanup() {
  if [ "$GENERATED_TASK_CONFIG" = true ] && [ -n "$TASK_CONFIG" ]; then
    rm -f "$TASK_CONFIG"
  fi
}
trap cleanup EXIT

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo-id) require_value "$1" "${2:-}"; REPO_ID="$2"; shift 2 ;;
    --exp) require_value "$1" "${2:-}"; EXP_NAME="$2"; shift 2 ;;
    --prompt) require_value "$1" "${2:-}"; PROMPT="$2"; shift 2 ;;
    --steps) require_value "$1" "${2:-}"; STEPS="$2"; shift 2 ;;
    --save-interval) require_value "$1" "${2:-}"; SAVE_INTERVAL="$2"; shift 2 ;;
    --batch-size) require_value "$1" "${2:-}"; BATCH_SIZE="$2"; shift 2 ;;
    --action-horizon) require_value "$1" "${2:-}"; ACTION_HORIZON="$2"; shift 2 ;;
    --rtc-delay) require_value "$1" "${2:-}"; RTC_DELAY="$2"; shift 2 ;;
    --delta) USE_DELTA=true; shift ;;
    --prompt-from-task) PROMPT_FROM_TASK=true; shift ;;
    --max-frames) require_value "$1" "${2:-}"; MAX_FRAMES="$2"; shift 2 ;;
    --skip-norm) SKIP_NORM=true; shift ;;
    --name) require_value "$1" "${2:-}"; NAME_OVERRIDE="$2"; shift 2 ;;
    --data-dir) require_value "$1" "${2:-}"; DATA_DIR="$2"; shift 2 ;;
    --weight) require_value "$1" "${2:-}"; WEIGHT_PATH="$2"; shift 2 ;;
    --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
    --task-config) require_value "$1" "${2:-}"; TASK_CONFIG="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --overwrite) OVERWRITE=true; shift ;;
    --wandb) USE_WANDB=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

require_positive_integer --steps "$STEPS"
require_positive_integer --save-interval "$SAVE_INTERVAL"
require_positive_integer --batch-size "$BATCH_SIZE"
require_positive_integer --action-horizon "$ACTION_HORIZON"
[ -z "$MAX_FRAMES" ] || require_positive_integer --max-frames "$MAX_FRAMES"
[ -z "$RTC_DELAY" ] || require_nonnegative_integer --rtc-delay "$RTC_DELAY"
[ "$RESUME" = false ] || [ "$OVERWRITE" = false ] || die "--resume and --overwrite are mutually exclusive"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR/logs" || die "cannot create output directory: $OUTPUT_DIR"

if [ -n "$TASK_CONFIG" ]; then
  [ -r "$TASK_CONFIG" ] || die "task config is not readable: $TASK_CONFIG"
  EXP_NAME="${EXP_NAME:-$(basename "$TASK_CONFIG")}"
  EXP_NAME="${EXP_NAME%.yaml}"
  EXP_NAME="${EXP_NAME%.yml}"
else
  [ -n "$REPO_ID" ] || die "--repo-id is required unless --task-config is used"
  [ -n "$EXP_NAME" ] || die "--exp is required unless --task-config is used"
  [[ "$REPO_ID" != /* ]] || die "--repo-id must be relative to --data-dir"
  case "/$REPO_ID/" in
    */../*|*/./*) die "--repo-id must not contain '.' or '..' path segments" ;;
  esac

  DATASET_ROOT="$DATA_DIR/$REPO_ID"
  [ -d "$DATASET_ROOT/data" ] || die "dataset data directory is missing: $DATASET_ROOT/data"
  [ -d "$DATASET_ROOT/meta" ] || die "dataset metadata directory is missing: $DATASET_ROOT/meta"
  [ -r "$WEIGHT_PATH" ] || die "model params are not readable: $WEIGHT_PATH"

  NAME="${NAME_OVERRIDE:-pi05_tron2_${EXP_NAME%%_*}}"
  [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "generated config name contains unsupported characters: $NAME"

  TASK_CONFIG="$(mktemp "${TMPDIR:-/tmp}/tron2_train_task.XXXXXX.yaml")"
  GENERATED_TASK_CONFIG=true

  if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_COMMAND=("$PYTHON_BIN")
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_COMMAND=(python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_COMMAND=(python)
  else
    die "python3 or python is required to generate the task config"
  fi

  "${PYTHON_COMMAND[@]}" - \
    "$TASK_CONFIG" "$NAME" "$REPO_ID" "$PROMPT" "$WEIGHT_PATH" \
    "$STEPS" "$SAVE_INTERVAL" "$BATCH_SIZE" "$ACTION_HORIZON" \
    "$USE_DELTA" "$PROMPT_FROM_TASK" "$RTC_DELAY" "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

(
    output_path,
    name,
    repo_id,
    prompt,
    weight_path,
    steps,
    save_interval,
    batch_size,
    action_horizon,
    use_delta,
    prompt_from_task,
    rtc_delay,
    output_dir,
) = sys.argv[1:]

config = {
    "name": name,
    "repo_id": repo_id,
    "prompt": prompt,
    "prompt_from_task": prompt_from_task == "true",
    "weight_loader": weight_path,
    "num_train_steps": int(steps),
    "save_interval": int(save_interval),
    "batch_size": int(batch_size),
    "action_horizon": int(action_horizon),
    "use_delta_joint_actions": use_delta == "true",
    "checkpoint_base_dir": output_dir,
    "assets_base_dir": str(pathlib.Path(output_dir) / "assets"),
}
if rtc_delay:
    config["rtc_training_simulated_delay"] = int(rtc_delay)

with pathlib.Path(output_path).open("w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)
    file.write("\n")
PY
fi

[[ "$EXP_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "experiment name contains unsupported characters: $EXP_NAME"

export HF_LEROBOT_HOME="$DATA_DIR"
if [ "$USE_WANDB" = false ]; then
  export WANDB_MODE=disabled
fi

NORM_COMMAND=(uv run scripts/compute_norm_stats.py --task-config "$TASK_CONFIG")
[ -z "$MAX_FRAMES" ] || NORM_COMMAND+=(--max-frames "$MAX_FRAMES")

TRAIN_COMMAND=(uv run scripts/train_tron2_task.py --task-config "$TASK_CONFIG" --exp-name "$EXP_NAME")
if [ "$RESUME" = true ]; then
  TRAIN_COMMAND+=(--resume)
elif [ "$OVERWRITE" = true ]; then
  TRAIN_COMMAND+=(--overwrite)
fi
[ "$USE_WANDB" = true ] || TRAIN_COMMAND+=(--no-wandb-enabled)

printf 'Project directory: %s\n' "$PROJECT_DIR"
printf 'Dataset root:     %s\n' "$DATA_DIR"
printf 'Model params:     %s\n' "$WEIGHT_PATH"
printf 'Output root:      %s\n' "$OUTPUT_DIR"
printf 'Task config:      %s\n' "$TASK_CONFIG"
cat "$TASK_CONFIG"
print_command 'Norm command:' "${NORM_COMMAND[@]}"
print_command 'Train command:' "${TRAIN_COMMAND[@]}"

if [ "$DRY_RUN" = true ]; then
  printf 'Dry run complete; training was not started.\n'
  exit 0
fi

command -v uv >/dev/null 2>&1 || die "uv is required; install dependencies before starting the job"

LOG_FILE="$OUTPUT_DIR/logs/training_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
if [ "$SKIP_NORM" = false ]; then
  printf '[%s] Computing normalization statistics...\n' "$(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
  "${NORM_COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

printf '[%s] Starting training...\n' "$(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"

printf 'Training complete. Checkpoints: %s/<config-name>/%s/<step>/\n' "$OUTPUT_DIR" "$EXP_NAME"
