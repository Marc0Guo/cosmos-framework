#!/usr/bin/env bash
# Continuous A40 experiment driver for HAMLET / runtime-adapt.
# Runs sequential LIBERO-10 LoRA smokes and appends a summary line each time.
set -euo pipefail
cd /workspace/cosmos-framework-hamlet
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source /workspace/run_cosmos3_framework.md 2>/dev/null || true
export LIBERO_ROOT="${LIBERO_ROOT:-/workspace/data/LIBERO_LeRobot_v3/libero_10}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-examples/checkpoints/Cosmos3-Nano}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
# Container root is tiny (~30G); keep HF/uv caches on /workspace.
export HF_HOME="${HF_HOME:-/workspace/caches/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/caches/uv}"
# Single A40 — launcher defaults to 8 ranks.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

NVIDIA_LIBS=$(python - <<'PY'
import site, pathlib
sp = pathlib.Path(site.getsitepackages()[0])
print(':'.join(str(p) for p in sorted(sp.glob('nvidia/*/lib')) if p.is_dir()))
PY
)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

LOG_ROOT=/workspace/outputs/hamlet_a40_loop
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
  echo -e "ts\trun_id\tcommit\tmean_loss\tlast10\tmin\tstatus\tdesc" > "$SUMMARY"
fi

mean_last() {
  local log="$1"
  python - <<PY
import re, statistics, sys
from pathlib import Path
vals = []
for path in Path("$log").parent.rglob("*.log"):
    text = path.read_text(errors="ignore")
    for m in re.finditer(r"Iteration\s+\d+:.*?Loss:\s*([0-9]+(?:\.[0-9]+)?)", text):
        vals.append(float(m.group(1)))
if not vals:
    # fallback: single train.log path
    text = Path("$log").read_text(errors="ignore")
    for m in re.finditer(r"Loss:\s*([0-9]+(?:\.[0-9]+)?)", text):
        vals.append(float(m.group(1)))
if not vals:
    print("nan\tnan\tnan")
    sys.exit(0)
last10 = vals[-10:] if len(vals) >= 10 else vals
print(f"{statistics.mean(vals):.6f}\t{statistics.mean(last10):.6f}\t{min(vals):.6f}")
PY
}

run_one() {
  local run_id="$1"
  local desc="$2"
  shift 2
  local out="$LOG_ROOT/$run_id"
  mkdir -p "$out"
  local commit
  commit=$(git rev-parse --short HEAD)
  echo "=== START $run_id commit=$commit $desc ===" | tee "$out/driver.log"
  export OUTPUT_ROOT="$out"
  export EXTRA_TAIL_OVERRIDES="$*"
  set +e
  bash examples/launch_sft_action_policy_libero_10_nano.sh >"$out/train.log" 2>&1
  local ec=$?
  set -e
  local metrics
  metrics=$(mean_last "$out/train.log")
  local status=keep
  [[ $ec -eq 0 ]] || status=crash
  echo -e "$(date -u +%Y-%m-%dT%H:%M:%SZ)\t${run_id}\t${commit}\t${metrics}\t${status}\t${desc}" | tee -a "$SUMMARY"
  echo "=== DONE $run_id exit=$ec metrics=$metrics ===" | tee -a "$out/driver.log"
  return $ec
}

COMMON="trainer.max_iter=50 job.wandb_mode=offline \
model.config.lora_enabled=true model.config.ema.enabled=false \
model.config.compile.enabled=false model.config.activation_checkpointing.mode=full \
model.config.max_num_tokens_after_packing=8192 \
model.config.parallelism.data_parallel_shard_degree=1 \
model.config.parallelism.data_parallel_replicate_degree=1 \
dataloader_train.max_samples_per_batch=1 trainer.logging_iter=1"

# E7: mem_dim=512 + real past-K (window=4)
run_one e7_mem512_pastk "hamlet mem512 past-K window=4" \
  $COMMON \
  model.config.hamlet.enabled=true \
  model.config.hamlet.memory_dim=512 \
  model.config.hamlet.memory_window=4 \
  model.config.hamlet.failure_buffer=false || true

# E8: + failure_buffer emphasize
run_one e8_mem512_failbuf "hamlet mem512 past-K + failure_buffer" \
  $COMMON \
  model.config.hamlet.enabled=true \
  model.config.hamlet.memory_dim=512 \
  model.config.hamlet.memory_window=4 \
  model.config.hamlet.failure_buffer=true \
  model.config.hamlet.failure_jump_threshold=0.35 || true

# E9: longer train with better of E7/E8 decided later — default failbuf on
run_one e9_mem512_failbuf_100 "hamlet mem512 failbuf max_iter=100" \
  trainer.max_iter=100 job.wandb_mode=offline \
  model.config.lora_enabled=true model.config.ema.enabled=false \
  model.config.compile.enabled=false model.config.activation_checkpointing.mode=full \
  model.config.max_num_tokens_after_packing=8192 \
  model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.parallelism.data_parallel_replicate_degree=1 \
  dataloader_train.max_samples_per_batch=1 trainer.logging_iter=1 \
  model.config.hamlet.enabled=true \
  model.config.hamlet.memory_dim=512 \
  model.config.hamlet.memory_window=4 \
  model.config.hamlet.failure_buffer=true \
  model.config.hamlet.failure_jump_threshold=0.35 || true

echo LOOP_DONE | tee -a "$SUMMARY"
cat "$SUMMARY"
