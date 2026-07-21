#!/bin/bash
# bench_fp8_kv_accuracy.sh — Accuracy validation: BF16 KV cache vs FP8 KV cache on Intel B70.
#
# Uses the RULER suite (NIAH variants + CWE + FWE + VT + QA) from lm-eval-harness
# to validate that FP8 KV cache preserves accuracy vs BF16 at 4K, 16K, and 32K context.
#
# BF16 is included as a canary: at long contexts, BF16 may OOM at server init.
# Those cases are recorded as "OOM" in the CSV, making the FP8 advantage explicit.
#
# OOM auto-retry: if a server fails to start AND extra_args does not already include
# --quantization fp8, the script automatically retries with it appended to reduce
# weight VRAM. If still OOM, the run is recorded as "OOM" and the script moves on.
#
# Models run SEQUENTIALLY (lm_eval is CPU-heavy; each eval takes ~30-60 min).
#
# Per-model canonical TP used for accuracy eval:
#   TP=1:  llama31, deepseekr1, gemma3, qwen3
#   TP=2:  gemma4, qwen25, mistral  (TP=1 OOMs at 32K for 14-24B models)
#   TP=4:  llama33_70b, qwen25_72b, deepseekr1_70b
#
# Usage:
#   bash bench_fp8_kv_accuracy.sh                          # all 10 models
#   bash bench_fp8_kv_accuracy.sh llama31 qwen25           # specific models
#   bash bench_fp8_kv_accuracy.sh --configs fp8 llama31    # fp8 KV only
#   bash bench_fp8_kv_accuracy.sh --phase 1                # TP=1 models only
#   bash bench_fp8_kv_accuracy.sh --phase 2                # TP=2 models only
#   bash bench_fp8_kv_accuracy.sh --phase 3                # TP=4 models only
#   bash bench_fp8_kv_accuracy.sh --contexts 4096,16384    # custom context lengths
#
# Output:
#   /tmp/fp8kv_accuracy/ruler_summary_<TS>.csv   (per-task scores; host: ~/LLM/fp8kv_accuracy/)
#   /tmp/ruler_summary_<TS>.csv                        (live host-visible copy)
set -euo pipefail

# ── Environment (required for Intel XPU backend) ────────────────────────────
export VLLM_TARGET_DEVICE=xpu
export VLLM_MLA_DISABLE=1
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT=900
export VLLM_NO_USAGE_STATS=1
export HF_HOME=/tmp   # container /tmp is bind-mounted to ~/LLM on host

BASE_PORT=8250
CONTEXT_LENGTHS=(4096 16384 32768)
CONFIGS=("bf16" "fp8")
RULER_TASKS="ruler"

RESULT_DIR="/tmp/fp8kv_accuracy"
LOG_DIR="${RESULT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV_SUMMARY="${RESULT_DIR}/ruler_summary_${TIMESTAMP}.csv"
# Host-visible copy: /tmp maps to /home/intel/LLM on the host
CSV_HOST="/tmp/ruler_summary_${TIMESTAMP}.csv"

ALL_MODELS=(llama31 deepseekr1 gemma3 qwen3 gemma4 qwen25 mistral llama33_70b qwen25_72b deepseekr1_70b gemma4_31b)

# ---------------------------------------------------------------------------
resolve_model() {
    case "$1" in
        llama31)        echo "meta-llama/Llama-3.1-8B-Instruct" ;;
        deepseekr1)     echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
        gemma3)         echo "google/gemma-3-1b-it" ;;
        qwen3)          echo "Qwen/Qwen3-8B" ;;
        gemma4)         echo "google/gemma-4-E4B-it" ;;
        qwen25)         echo "Qwen/Qwen2.5-14B-Instruct" ;;
        mistral)        echo "mistralai/Mistral-Small-24B-Instruct-2501" ;;
        llama33_70b)    echo "meta-llama/Llama-3.3-70B-Instruct" ;;
        qwen25_72b)     echo "Qwen/Qwen2.5-72B-Instruct" ;;
        deepseekr1_70b) echo "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" ;;
        gemma4_31b)     echo "google/gemma-4-31B-it" ;;
        *)              echo "UNKNOWN"; return 1 ;;
    esac
}

# Single canonical TP for accuracy evaluation (not all TPs — accuracy doesn't
# change with TP, but OOM risk does, so use the safest TP for long context).
resolve_tp() {
    case "$1" in
        llama33_70b|qwen25_72b|deepseekr1_70b) echo "4" ;;
        gemma4|qwen25|mistral|gemma4_31b)       echo "2" ;;
        *)                                      echo "1" ;;
    esac
}

# Extra vllm serve flags per model.
resolve_extra_args() {
    case "$1" in
        deepseekr1|deepseekr1_70b) echo "--trust-remote-code --quantization fp8" ;;
        gemma4)         echo "--trust-remote-code --attention-backend FLASH_ATTN --quantization fp8" ;;
        gemma4_31b)     echo "--trust-remote-code --attention-backend FLASH_ATTN --quantization fp8" ;;
        *)              echo "--quantization fp8" ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse flags: --configs  --phase  --contexts
PHASE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs)
            IFS=',' read -ra CONFIGS <<< "$2"; shift 2 ;;
        --phase)
            PHASE="$2"; shift 2 ;;
        --contexts)
            IFS=',' read -ra CONTEXT_LENGTHS <<< "$2"; shift 2 ;;
        *) break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    set -- "${ALL_MODELS[@]}"
fi
MODEL_SHORTS=("$@")
for ms in "${MODEL_SHORTS[@]}"; do
    if ! resolve_model "${ms}" > /dev/null 2>&1; then
        echo "Unknown model: ${ms}. Choose from: ${ALL_MODELS[*]}"; exit 1
    fi
done

# Apply --phase filter based on canonical TP
if [[ -n "${PHASE}" ]]; then
    FILTERED=()
    for ms in "${MODEL_SHORTS[@]}"; do
        tp=$(resolve_tp "${ms}")
        case "${PHASE}" in
            1) [[ "${tp}" == "1" ]] && FILTERED+=("${ms}") ;;
            2) [[ "${tp}" == "2" ]] && FILTERED+=("${ms}") ;;
            3) [[ "${tp}" == "4" ]] && FILTERED+=("${ms}") ;;
        esac
    done
    MODEL_SHORTS=("${FILTERED[@]+"${FILTERED[@]}"}")
    if [[ ${#MODEL_SHORTS[@]} -eq 0 ]]; then
        echo "No models match --phase ${PHASE}."; exit 0
    fi
fi

# ---------------------------------------------------------------------------
start_server() {
    local model_short="$1" config="$2" model="$3" extra_args="$4" card="$5" port="$6" tp="$7" ctx_len="$8"
    local server_log="${LOG_DIR}/vllm_ruler_server_${model_short}_${config}_tp${tp}_ctx${ctx_len}.log"

    echo "[${model_short}] Starting: config=${config} tp=${tp} card=${card} port=${port} ctx=${ctx_len}"

    local kv_arg=""
    [[ "${config}" != "bf16" ]] && kv_arg="--kv-cache-dtype ${config}"

    # --max-num-batched-tokens deliberately omitted from vllm serve below.
    # Scaling it with ctx_len (or even fixing it high) forces vLLM to profile
    # activation memory for a giant-batch forward pass, starving KV cache at
    # long context (e.g. OOMs at 32K on gemma4_31b TP=2 even though vLLM's own
    # default lets the same model fit 64K on the same B70x2 hardware via
    # chunked prefill). Let vLLM pick its own default instead.

    # 70B models: skip ZE_AFFINITY_MASK and --dtype bfloat16 (let vLLM auto-detect).
    local ze_prefix=""
    local dtype_arg="--dtype bfloat16"
    if [[ "${model_short}" != "llama33_70b" && "${model_short}" != "qwen25_72b" && "${model_short}" != "deepseekr1_70b" ]]; then
        local affinity_mask="${card}"
        if [[ "${tp}" == "2" ]]; then
            affinity_mask="${card},$(( card + 1 ))"
        fi
        ze_prefix="ZE_AFFINITY_MASK=${affinity_mask}"
    else
        dtype_arg=""
    fi

    local serve_cmd="${ze_prefix} VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
        vllm serve ${model} \
        --port ${port} \
        ${dtype_arg} \
        --tensor-parallel-size ${tp} \
        --max-model-len ${ctx_len} \
        --gpu-memory-utilization 0.92 \
        --enforce-eager \
        --max-num-seqs 2 \
        --block-size 64 \
        --no-enable-log-requests \
        --no-enable-prefix-caching \
        ${kv_arg} \
        ${extra_args} \
        > ${server_log} 2>&1"

    setsid bash -c "${serve_cmd}" &
    local server_pid=$!
    echo "${server_pid}" > "${LOG_DIR}/vllm_ruler_pid_${model_short}_${port}.txt"

    echo "[${model_short}]   Waiting for server (up to 900s)..."
    local attempts=0
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        attempts=$(( attempts + 1 ))
        if [[ ${attempts} -gt 900 ]]; then
            echo "[${model_short}]   ERROR: server failed to start after 900s"
            tail -40 "${server_log}" 2>/dev/null || true
            stop_server "${model_short}" "${port}"
            return 1
        fi
        if [[ $(( attempts % 30 )) -eq 0 ]]; then
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                echo "[${model_short}]   OOM or crash at init (config=${config} tp=${tp} ctx=${ctx_len})"
                tail -20 "${server_log}" 2>/dev/null | grep -iE "error|oom|memory|killed" || true
                return 1
            fi
        fi
        sleep 1
    done
    echo "[${model_short}]   Server ready in ${attempts}s"
}

stop_server() {
    local model_short="$1" port="$2"
    echo "[${model_short}]   Stopping server on port ${port}..."
    local pid_file="${LOG_DIR}/vllm_ruler_pid_${model_short}_${port}.txt"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        local pgid
        pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)
        if [[ -n "${pgid}" && "${pgid}" != "0" ]]; then
            kill -- "-${pgid}" 2>/dev/null || true
        fi
        rm -f "${pid_file}"
    fi
    pkill -TERM -f "vllm serve.*--port ${port}" 2>/dev/null || true
    sleep 3
    pkill -KILL -f "vllm serve.*--port ${port}" 2>/dev/null || true
    # Do NOT use global pkill for EngineCore/spawn — it would kill other parallel models.
    # The setsid+pgid kill above already cleans up all child processes for this server.
    sleep 3
    local i=0
    while ss -tln 2>/dev/null | grep -q ":${port} " && [[ ${i} -lt 30 ]]; do
        sleep 1; i=$(( i + 1 ))
    done
}

# ---------------------------------------------------------------------------
run_ruler() {
    local model_short="$1" config="$2" model="$3" port="$4" tp="$5" ctx_len="$6"
    local output_dir="${RESULT_DIR}/ruler_${model_short}_${config}_ctx${ctx_len}_${TIMESTAMP}"
    local eval_log="${LOG_DIR}/ruler_${model_short}_${config}_ctx${ctx_len}.log"

    # Use fewer samples for large/slow models to keep eval tractable
    local limit=50
    [[ $(( tp )) -ge 2 ]] && limit=25

    echo "[${model_short}]   Running RULER (ctx=${ctx_len} config=${config} limit=${limit})..."

    python3 -c "
import sys, json, os

import lm_eval.tasks.ruler.common_utils as cu
cu.DEFAULT_SEQ_LENGTHS = [${ctx_len}]

from lm_eval import evaluator

model_args = {
    'base_url': 'http://localhost:${port}/v1/completions',
    'model': '${model}',
    'tokenizer_backend': 'huggingface',
    'num_concurrent': 1,
    'max_retries': 5,
    'timeout': 1800,
}

results = evaluator.simple_evaluate(
    model='local-completions',
    model_args=model_args,
    tasks='${RULER_TASKS}'.split(','),
    metadata={'max_seq_lengths': [${ctx_len}], 'tokenizer': '${model}'},
    batch_size='auto',
    log_samples=True,
    limit=${limit},
)

output_dir = '${output_dir}'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results.get('results', {}), indent=2, default=str))
" > "${eval_log}" 2>&1
    local rc=$?

    if [[ ${rc} -ne 0 ]]; then
        echo "[${model_short}]     RULER FAILED (rc=${rc})"
        tail -30 "${eval_log}" 2>/dev/null || true
        return 1
    fi

    echo "[${model_short}]     RULER complete — results in ${output_dir}"

    # Extract per-task scores and append to raw CSV
    python3 -c "
import json, os

output_dir = '${output_dir}'
results_file = os.path.join(output_dir, 'results.json')
if not os.path.exists(results_file):
    print('No results file found', file=__import__('sys').stderr)
    exit(1)

with open(results_file) as f:
    data = json.load(f)

results = data.get('results', {})
for task_name, metrics in results.items():
    if task_name in ('alias',):
        continue
    for key, val in metrics.items():
        # lm-eval formats keys as '4096,none' or '4096_stderr,none'
        # Extract the numeric context length from the prefix before the comma
        if not isinstance(key, str) or '_stderr' in key or key == 'alias':
            continue
        ctx_str = key.split(',')[0].strip()
        if not ctx_str.isdigit():
            continue
        ctx = int(ctx_str)
        if not isinstance(val, (int, float)) or val < 0:
            continue
        print(f'${model_short},${config},${tp},{ctx},{task_name},{val:.4f}')
" 2>/dev/null | tee -a "${CSV_SUMMARY}" >> "${CSV_HOST}" || true
}

# ---------------------------------------------------------------------------
run_model_suite() {
    local model_short="$1" card="${2:-0}" port="${3:-${BASE_PORT}}"
    local model extra_args tp
    model=$(resolve_model "${model_short}")
    extra_args=$(resolve_extra_args "${model_short}")
    tp=$(resolve_tp "${model_short}")

    echo ""
    echo "[${model_short}] ========================================"
    echo "[${model_short}]  Model:  ${model}"
    echo "[${model_short}]  TP=${tp}  Card=${card}  Port=${port}"
    echo "[${model_short}] ========================================"

    for ctx_len in "${CONTEXT_LENGTHS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            echo ""
            echo "[${model_short}] --- ctx=${ctx_len} config=${config} ---"

            local server_rc=0
            start_server "${model_short}" "${config}" "${model}" \
                "${extra_args}" "${card}" "${port}" "${tp}" "${ctx_len}" || server_rc=$?

            if [[ ${server_rc} -ne 0 ]]; then
                # OOM retry: add --quantization fp8 to halve weight VRAM
                if [[ "${extra_args}" != *"--quantization fp8"* ]]; then
                    echo "[${model_short}]   Retrying with --quantization fp8 (weights) to reduce VRAM..."
                    stop_server "${model_short}" "${port}"
                    local retry_rc=0
                    start_server "${model_short}" "${config}" "${model}" \
                        "${extra_args} --quantization fp8" \
                        "${card}" "${port}" "${tp}" "${ctx_len}" || retry_rc=$?
                    if [[ ${retry_rc} -ne 0 ]]; then
                        echo "[${model_short}]   Still OOM — recording OOM"
                        _oom_row="${model_short},${config},${tp},${ctx_len},ALL,OOM"
                        echo "${_oom_row}" | tee -a "${CSV_SUMMARY}" >> "${CSV_HOST}"
                        continue
                    fi
                    echo "[${model_short}]   Retry succeeded with --quantization fp8"
                else
                    echo "[${model_short}]   OOM — recording OOM"
                    _oom_row="${model_short},${config},${tp},${ctx_len},ALL,OOM"
                    echo "${_oom_row}" | tee -a "${CSV_SUMMARY}" >> "${CSV_HOST}"
                    continue
                fi
            fi

            local eval_rc=0
            run_ruler "${model_short}" "${config}" "${model}" "${port}" "${tp}" "${ctx_len}" \
                || eval_rc=$?

            if [[ ${eval_rc} -ne 0 ]]; then
                echo "[${model_short}]   Retrying eval once after server restart..."
                stop_server "${model_short}" "${port}"
                sleep 5
                local retry_rc2=0
                start_server "${model_short}" "${config}" "${model}" \
                    "${extra_args}" "${card}" "${port}" "${tp}" "${ctx_len}" || retry_rc2=$?
                if [[ ${retry_rc2} -eq 0 ]]; then
                    run_ruler "${model_short}" "${config}" "${model}" "${port}" "${tp}" "${ctx_len}" || \
                        { _u="${model_short},${config},${tp},${ctx_len},ALL,UNSTABLE"; echo "${_u}" | tee -a "${CSV_SUMMARY}" >> "${CSV_HOST}"; }
                else
                    _u="${model_short},${config},${tp},${ctx_len},ALL,UNSTABLE"
                    echo "${_u}" | tee -a "${CSV_SUMMARY}" >> "${CSV_HOST}"
                fi
            fi

            stop_server "${model_short}" "${port}"
        done
    done
}

# ---------------------------------------------------------------------------
# Unified resource-aware scheduler
#
# All models (any TP mix) go into a single pending queue.
# A scheduler loop polls every 5s and greedily dispatches each pending job
# to the first available card group that fits its TP:
#   TP=1 → any single free card  (port BASE_PORT+card)
#   TP=2 → cards 0,1 or cards 2,3  (port BASE_PORT+slot)
#   TP=4 → all 4 cards  (port BASE_PORT)
#
# Card locks use mkdir (atomic on Linux).  Workers run in the background
# and release their card locks on completion — so as soon as a TP=2 pair
# of cards frees up, the next TP=2 job is dispatched immediately.
# ---------------------------------------------------------------------------

LOCK_DIR="/tmp/ruler_locks_${TIMESTAMP}"

# Atomically try to lock all given card numbers.
# Returns 0 on success; rolls back and returns 1 if any card is busy.
try_lock_cards() {
    local acquired=()
    for c in "$@"; do
        if mkdir "${LOCK_DIR}/card_${c}" 2>/dev/null; then
            acquired+=("${c}")
        else
            for a in "${acquired[@]+${acquired[@]}}"; do
                rmdir "${LOCK_DIR}/card_${a}" 2>/dev/null || true
            done
            return 1
        fi
    done
    return 0
}

release_cards() {
    for c in "$@"; do
        rmdir "${LOCK_DIR}/card_${c}" 2>/dev/null || true
    done
}

# Background worker: runs one model suite then releases its cards.
_run_worker() {
    local model_short="$1" card="$2" port="$3"; shift 3
    local rel_cards=("$@")
    run_model_suite "${model_short}" "${card}" "${port}" || true
    release_cards "${rel_cards[@]}"
}

run_scheduler() {
    local models=("$@")
    [[ ${#models[@]} -eq 0 ]] && return 0
    mkdir -p "${LOCK_DIR}"

    local pending=("${models[@]}")
    local pids=()

    echo ""
    echo ">>> Scheduler: ${#pending[@]} jobs queued — dispatching as cards become free"
    sleep 3

    while [[ ${#pending[@]} -gt 0 || ${#pids[@]} -gt 0 ]]; do
        # Reap finished workers
        local alive=()
        for pid in "${pids[@]+${pids[@]}}"; do
            kill -0 "${pid}" 2>/dev/null && alive+=("${pid}") || true
        done
        pids=("${alive[@]+${alive[@]}}")

        # Try to dispatch each pending job in queue order
        local still_pending=()
        local scheduled_any=0
        for ms in "${pending[@]+${pending[@]}}"; do
            local tp
            tp=$(resolve_tp "${ms}")
            local dispatched=0

            if [[ "${tp}" == "4" ]]; then
                if try_lock_cards 0 1 2 3; then
                    echo ">>> [$(date +%H:%M:%S)] Dispatch ${ms} (TP=4) cards=0-3 port=${BASE_PORT}"
                    _run_worker "${ms}" "0" "${BASE_PORT}" 0 1 2 3 &
                    pids+=($!)
                    dispatched=1; scheduled_any=1
                fi
            elif [[ "${tp}" == "2" ]]; then
                if try_lock_cards 0 1; then
                    local p=${BASE_PORT}
                    echo ">>> [$(date +%H:%M:%S)] Dispatch ${ms} (TP=2) cards=0,1 port=${p}"
                    _run_worker "${ms}" "0" "${p}" 0 1 &
                    pids+=($!)
                    dispatched=1; scheduled_any=1
                elif try_lock_cards 2 3; then
                    local p=$(( BASE_PORT + 1 ))
                    echo ">>> [$(date +%H:%M:%S)] Dispatch ${ms} (TP=2) cards=2,3 port=${p}"
                    _run_worker "${ms}" "2" "${p}" 2 3 &
                    pids+=($!)
                    dispatched=1; scheduled_any=1
                fi
            else  # TP=1
                for c in 0 1 2 3; do
                    if try_lock_cards "${c}"; then
                        local p=$(( BASE_PORT + c ))
                        echo ">>> [$(date +%H:%M:%S)] Dispatch ${ms} (TP=1) card=${c} port=${p}"
                        _run_worker "${ms}" "${c}" "${p}" "${c}" &
                        pids+=($!)
                        dispatched=1; scheduled_any=1
                        break
                    fi
                done
            fi

            [[ ${dispatched} -eq 0 ]] && still_pending+=("${ms}")
        done
        pending=("${still_pending[@]+${still_pending[@]}}")

        # Sleep only when all cards are busy and work remains
        if [[ ${#pending[@]} -gt 0 && ${scheduled_any} -eq 0 ]]; then
            sleep 5
        fi
    done

    # Final wait for any stragglers
    for pid in "${pids[@]+${pids[@]}}"; do
        wait "${pid}" || FAIL=1
    done
    rm -rf "${LOCK_DIR}"
    echo ">>> Scheduler complete"
}

# ---------------------------------------------------------------------------
emit_summary() {
    echo ""
    echo "================================================================"
    echo "  RULER Accuracy: BF16 KV vs FP8 KV Cache"
    echo "================================================================"
    cat "${CSV_SUMMARY}"

    echo ""
    echo "--- Per-model BF16 vs FP8 KV accuracy delta ---"
    python3 - "${CSV_SUMMARY}" <<'PY' 2>/dev/null || true
import sys, csv
from collections import defaultdict

csv_file = sys.argv[1]
data = defaultdict(dict)
try:
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['model'], row['tp'], row['context_length'], row['task'])
            data[key][row['config']] = row['score']
except Exception as e:
    print(f"Could not read CSV: {e}", file=sys.stderr)
    sys.exit(0)

print(f"{'Model':<14} {'TP':>3} {'Ctx':>6} {'Task':<22} {'BF16':>8} {'FP8-KV':>8} {'Delta':>9}")
print("-" * 78)
for (model, tp, ctx, task), scores in sorted(data.items()):
    bf16 = scores.get('bf16', 'N/A')
    fp8  = scores.get('fp8',  'N/A')
    delta = ''
    try:
        if bf16 not in ('N/A', 'OOM', 'UNSTABLE') and fp8 not in ('N/A', 'OOM', 'UNSTABLE'):
            delta = f"{float(fp8) - float(bf16):+.4f}"
        elif bf16 in ('OOM', 'UNSTABLE') and fp8 not in ('N/A', 'OOM', 'UNSTABLE'):
            delta = f"BF16 {bf16}"
    except ValueError:
        pass
    print(f"{model:<14} {tp:>3} {ctx:>6} {task:<22} {bf16:>8} {fp8:>8} {delta:>9}")
PY

    echo ""
    echo "Results: ${CSV_SUMMARY}"
    echo "Host:    ${CSV_HOST}"
}

# ===========================================================================
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# Write CSV header once at startup — rows appended live after each eval
HEADER="model,config,tp,context_length,task,score"
echo "${HEADER}" > "${CSV_SUMMARY}"
echo "${HEADER}" > "${CSV_HOST}"

# ---------------------------------------------------------------------------
# Dependency check / install
# ---------------------------------------------------------------------------
echo "Checking Python dependencies..."
python3 - <<'DEPCHECK'
import sys, subprocess

REQUIRED = {
    "lm_eval":          "lm-eval[api]",
    "nltk":             "nltk",
    "rouge_score":      "rouge-score",
    "sklearn":          "scikit-learn",
    "sentencepiece":    "sentencepiece",
    "tiktoken":         "tiktoken",
}

missing = []
for module, pkg in REQUIRED.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"  Installing missing packages: {' '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
    print("  Done.")
else:
    print("  All dependencies present.")

# Verify lm_eval has RULER tasks
try:
    import lm_eval.tasks.ruler.common_utils  # noqa
    print("  RULER tasks: OK")
except ImportError:
    print("  WARNING: RULER tasks not found in lm_eval. Install lm-eval from source:")
    print("    pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git")
    sys.exit(1)

# Download required NLTK data silently
import nltk
for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        print(f"  Downloading NLTK data: {pkg}")
        nltk.download(pkg, quiet=True)
DEPCHECK
# ---------------------------------------------------------------------------

# Print banner
ACTIVE_MODELS_STR="${MODEL_SHORTS[*]}"
echo "============================================"
echo "  FP8 KV Cache Accuracy Benchmark (RULER)"
[[ -n "${PHASE}" ]] && echo "  Phase:    ${PHASE}"
echo "  Models:   ${ACTIVE_MODELS_STR}"
echo "  Configs:  ${CONFIGS[*]}"
echo "  Context:  ${CONTEXT_LENGTHS[*]}"
echo "  Tasks:    ${RULER_TASKS}"
echo "  Timestamp: ${TIMESTAMP}"
echo "  BF16 included as canary (may OOM at long context)"
echo "  Scheduler: cards claimed dynamically — TP=1 on any free card,"
echo "             TP=2 on first free pair (0,1 or 2,3), TP=4 when all free"
echo "============================================"

# Check for leftover vllm processes
echo "Checking for leftover vllm processes..."
if pgrep -f "vllm serve" > /dev/null 2>&1; then
    echo "  Found — killing..."
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    pkill -KILL -f "from multiprocessing.spawn" 2>/dev/null || true
    sleep 3
    echo "  Done."
else
    echo "  None found. Proceeding."
fi

FAIL=0
run_scheduler "${MODEL_SHORTS[@]}"

emit_summary
exit ${FAIL}
