#!/bin/bash
# bench_tq_long_context_accuracy.sh — Long-context accuracy with RULER suite.
#
# Validates that TurboQuant preserves accuracy at 16K and 32K context on B70.
# Uses the RULER suite (NIAH variants + CWE + FWE + VT + QA) from lm-eval-harness.
#
# BF16 is included as canary: if the server OOMs, accuracy is recorded as "OOM".
# This forecloses the reviewer question "is NIAH alone enough?"
#
# Per-model TP: Gemma-4-E4B and Llama-3.1-8B use TP=1, Qwen2.5-14B uses TP=2.
# Models run SEQUENTIALLY (lm_eval is CPU-heavy, and each eval takes ~30-60min).
#
# Usage:
#   bash bench_tq_long_context_accuracy.sh                    # all 3 models
#   bash bench_tq_long_context_accuracy.sh llama31_8b         # one model
#
# Output:
#   /workspace/bench_results/ruler_<model>_<config>_<ctx>_<TS>/  — lm_eval output dirs
#   /workspace/bench_results/ruler_summary_<TS>.csv              — aggregated CSV
set -euo pipefail

CONTAINER="vllm-test"
BASE_PORT=8230
CONTEXT_LENGTHS=(4096 16384 32768)

# BF16 baseline + TQ preset
CONFIGS=("bf16" "turboquant_4bit_nc")

# Full RULER suite: NIAH variants + common-word extraction + freq-word extraction
# + variable tracking + QA. Skip individual niah_* (they're in the ruler group).
RULER_TASKS="ruler"

RESULT_DIR="/workspace/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV_SUMMARY="${RESULT_DIR}/ruler_summary_${TIMESTAMP}.csv"

# ---------------------------------------------------------------------------
resolve_model() {
    case "$1" in
        gemma4_e4b)  echo "google/gemma-4-E4B-it" ;;
        llama31_8b) echo "meta-llama/Llama-3.1-8B-Instruct" ;;
        qwen25_14b) echo "Qwen/Qwen2.5-14B-Instruct" ;;
        *)          echo "UNKNOWN"; return 1 ;;
    esac
}

resolve_tp() {
    case "$1" in
        gemma4_e4b)  echo 1 ;;
        llama31_8b) echo 1 ;;
        qwen25_14b) echo 2 ;;
        *)          echo 2 ;;
    esac
}

resolve_gpu() {
    case "$1" in
        gemma4_e4b)  echo "0" ;;
        llama31_8b) echo "1" ;;
        *)          echo "0,1" ;;
    esac
}

resolve_port() {
    case "$1" in
        gemma4_e4b)  echo $((BASE_PORT)) ;;
        llama31_8b) echo $((BASE_PORT + 1)) ;;
        qwen25_14b) echo $((BASE_PORT + 2)) ;;
        *)          echo $((BASE_PORT)) ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse optional --configs flag (comma-separated, e.g. --configs bf16)
CONFIG_OVERRIDE=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs) CONFIG_OVERRIDE="$2"; shift 2 ;;
        *)         POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]:-}"
[[ -n "${CONFIG_OVERRIDE}" ]] && IFS=',' read -ra CONFIGS <<< "${CONFIG_OVERRIDE}"

if [[ $# -eq 0 ]]; then
    set -- gemma4_e4b llama31_8b qwen25_14b
fi
MODEL_SHORTS=("$@")
for ms in "${MODEL_SHORTS[@]}"; do
    if ! resolve_model "${ms}" > /dev/null 2>&1; then
        echo "Unknown model: ${ms}. Choose from: gemma4_e4b, llama31_8b, qwen25_14b"; exit 1
    fi
done

# ---------------------------------------------------------------------------
start_server() {
    local model_short="$1" config="$2" model="$3" tp_size="$4" gpu="$5" port="$6" ctx_len="$7"
    local server_log="/tmp/vllm_ruler_server_${model_short}.log"

    echo "[${model_short}] Starting: config=${config} TP=${tp_size} GPU=${gpu} port=${port} ctx=${ctx_len}"

    local kv_arg=""
    [[ "${config}" != "bf16" ]] && kv_arg="--kv-cache-dtype ${config}"

    # BF16 TP=2 is prone to mid-inference EngineCore crashes under sustained load;
    # lower utilization leaves headroom and reduces crash probability.
    local gpu_mem_util="0.92"
    [[ "${tp_size}" -ge 2 && "${config}" == "bf16" ]] && gpu_mem_util="0.70"

    local max_batched=${ctx_len}
    [[ ${max_batched} -lt 8192 ]] && max_batched=8192

    local serve_cmd="ZE_AFFINITY_MASK=${gpu} \
        VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
        vllm serve ${model} \
        --port ${port} \
        --tensor-parallel-size ${tp_size} \
        --dtype bfloat16 \
        --max-model-len ${ctx_len} \
        --gpu-memory-utilization ${gpu_mem_util} \
        --enforce-eager \
        --max-num-batched-tokens ${max_batched} \
        --block-size 64 \
        --no-enable-log-requests \
        --no-enable-prefix-caching \
        --max-num-seqs 2 \
        ${kv_arg} \
        > ${server_log} 2>&1"

    docker exec -d ${CONTAINER} bash -c "${serve_cmd}"

    local attempts=0
    while ! docker exec ${CONTAINER} curl -sf http://localhost:${port}/health > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ ${attempts} -gt 600 ]]; then
            if ! docker exec ${CONTAINER} pgrep -f "vllm serve.*--port ${port}" > /dev/null 2>&1; then
                echo "[${model_short}]   OOM or crash at init (config=${config} ctx=${ctx_len})"
                docker exec ${CONTAINER} tail -20 "${server_log}" 2>/dev/null | grep -iE "error|oom|memory|killed" || true
                stop_server "${model_short}" "${port}"
                return 1
            fi
            echo "[${model_short}]   ERROR: server failed to start (timeout)"
            docker exec ${CONTAINER} tail -40 "${server_log}" 2>/dev/null || true
            stop_server "${model_short}" "${port}"
            return 1
        fi
        if [[ $((attempts % 30)) -eq 0 ]]; then
            if ! docker exec ${CONTAINER} pgrep -f "vllm serve.*--port ${port}" > /dev/null 2>&1; then
                echo "[${model_short}]   OOM or crash at init (config=${config} ctx=${ctx_len})"
                docker exec ${CONTAINER} tail -20 "${server_log}" 2>/dev/null | grep -iE "error|oom|memory|killed" || true
                stop_server "${model_short}" "${port}"
                return 1
            fi
        fi
        sleep 1
    done
    echo "[${model_short}]   Server ready in ${attempts}s"
}

stop_server() {
    local model_short="$1" port="$2" tp_size="${3:-1}"
    echo "[${model_short}]   Stopping server on port ${port}..."
    docker exec ${CONTAINER} bash -c "
        pkill -TERM -f 'vllm serve.*--port ${port}' 2>/dev/null || true
        sleep 3
        pkill -KILL -f 'vllm serve.*--port ${port}' 2>/dev/null || true
        sleep 2
        pkill -KILL -f 'EngineCore' 2>/dev/null || true
        pkill -KILL -f 'from multiprocessing.spawn' 2>/dev/null || true
        pkill -KILL -f 'VLLM::Wor' 2>/dev/null || true
        rm -f /dev/shm/psm_* /dev/shm/sem.loky-* 2>/dev/null || true
        for i in \$(seq 1 30); do
            ss -tln 2>/dev/null | grep -q ':${port} ' || break
            sleep 1
        done
    " || true
    sleep 5
    # TP=2 workers spawn on the host and survive container-level kills.
    # docker restart is the only reliable way to reclaim leaked GPU memory.
    if [[ "${tp_size}" -ge 2 ]]; then
        echo "[${model_short}]   TP=2: restarting container to reclaim host GPU memory..."
        docker restart ${CONTAINER} > /dev/null 2>&1 || true
        sleep 15
        docker exec ${CONTAINER} mkdir -p "${RESULT_DIR}" 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
run_ruler() {
    local model_short="$1" config="$2" model="$3" port="$4" ctx_len="$5"
    local output_dir="${RESULT_DIR}/ruler_${model_short}_${config}_ctx${ctx_len}_${TIMESTAMP}"
    local eval_log="/tmp/ruler_${model_short}_${config}_ctx${ctx_len}.log"

    # TP=2 model is slow — use fewer samples to stay tractable
    local limit_note=""
    [[ "${model_short}" == "qwen25_14b" ]] && limit_note=" limit=25"

    echo "[${model_short}]   Running RULER (ctx=${ctx_len}, config=${config}${limit_note})..."

    # We need to:
    # 1. Patch DEFAULT_SEQ_LENGTHS so process_results creates the right metric keys
    # 2. Pass max_seq_lengths so dataset generation creates the right context lengths
    # 3. Pass tokenizer so RULER can tokenize haystack to target length
    docker exec ${CONTAINER} python3 -c "
import sys, json, os

# Patch DEFAULT_SEQ_LENGTHS before task loading
import lm_eval.tasks.ruler.common_utils as cu
cu.DEFAULT_SEQ_LENGTHS = [${ctx_len}]

from lm_eval import evaluator

model_args = {
    'base_url': 'http://localhost:${port}/v1/completions',
    'model': '${model}',
    'tokenizer_backend': 'huggingface',
    'num_concurrent': 1,
    'max_retries': 5,
    'timeout': 1200,
}

limit = 25 if '${model_short}' == 'qwen25_14b' else None

results = evaluator.simple_evaluate(
    model='local-completions',
    model_args=model_args,
    tasks='${RULER_TASKS}'.split(','),
    metadata={'max_seq_lengths': [${ctx_len}], 'tokenizer': '${model}'},
    batch_size='auto',
    log_samples=True,
    limit=limit,
)

# Save results manually
output_dir = '${output_dir}'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Print summary
print(json.dumps(results.get('results', {}), indent=2, default=str))
" > "${eval_log}" 2>&1
    local rc=$?

    if [[ ${rc} -ne 0 ]]; then
        echo "[${model_short}]     RULER FAILED (rc=${rc})"
        tail -30 "${eval_log}" 2>/dev/null || true
        return 1
    fi

    echo "[${model_short}]     RULER complete — results in ${output_dir}"
    # Extract scores and append to CSV
    docker exec ${CONTAINER} python3 -c "
import json, os

output_dir = '${output_dir}'
results_file = os.path.join(output_dir, 'results.json')
if not os.path.exists(results_file):
    print('No results file found')
    exit(1)

with open(results_file) as f:
    data = json.load(f)

results = data.get('results', {})
for task_name, metrics in results.items():
    # Each task has metric keys like '4096', '16384', etc.
    for key, val in metrics.items():
        if key.isdigit():
            ctx = int(key)
            score = val if isinstance(val, (int, float)) else -1
            if score == -1:
                continue
            print(f'${model_short},${config},{ctx},{task_name},{score:.4f}')
" 2>/dev/null >> "/tmp/ruler_raw_${TIMESTAMP}.csv" || true
}

# ---------------------------------------------------------------------------
run_model_suite() {
    local model_short="$1"
    local model tp_size gpu port
    model=$(resolve_model "${model_short}")
    tp_size=$(resolve_tp "${model_short}")
    gpu=$(resolve_gpu "${model_short}")
    port=$(resolve_port "${model_short}")

    echo ""
    echo "[${model_short}] ============================================"
    echo "[${model_short}]   Model: ${model}  TP=${tp_size}  GPU=${gpu}  port=${port}"
    echo "[${model_short}] ============================================"

    for ctx_len in "${CONTEXT_LENGTHS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            echo ""
            echo "[${model_short}] --- ctx=${ctx_len} config=${config} ---"
            if start_server "${model_short}" "${config}" "${model}" "${tp_size}" "${gpu}" "${port}" "${ctx_len}"; then
                if ! run_ruler "${model_short}" "${config}" "${model}" "${port}" "${ctx_len}"; then
                    echo "[${model_short}]   Evaluation failed — restarting server for retry..."
                    stop_server "${model_short}" "${port}" "${tp_size}"
                    sleep 5
                    if start_server "${model_short}" "${config}" "${model}" "${tp_size}" "${gpu}" "${port}" "${ctx_len}"; then
                        run_ruler "${model_short}" "${config}" "${model}" "${port}" "${ctx_len}" || \
                            echo "${model_short},${config},${ctx_len},ALL,UNSTABLE" >> "/tmp/ruler_raw_${TIMESTAMP}.csv"
                    else
                        echo "[${model_short}]   Retry server start failed — recording UNSTABLE"
                        echo "${model_short},${config},${ctx_len},ALL,UNSTABLE" \
                            >> "/tmp/ruler_raw_${TIMESTAMP}.csv"
                    fi
                fi
            else
                echo "[${model_short}]   Server failed to start — recording OOM"
                echo "${model_short},${config},${ctx_len},ALL,OOM" \
                    >> "/tmp/ruler_raw_${TIMESTAMP}.csv"
            fi
            stop_server "${model_short}" "${port}" "${tp_size}"
        done
    done
}

# ---------------------------------------------------------------------------
emit_summary() {
    docker exec ${CONTAINER} mkdir -p "${RESULT_DIR}"

    # Build summary CSV
    docker exec ${CONTAINER} bash -c "echo 'model,config,context_length,task,score' > ${CSV_SUMMARY}"
    if [[ -f "/tmp/ruler_raw_${TIMESTAMP}.csv" ]]; then
        docker cp "/tmp/ruler_raw_${TIMESTAMP}.csv" "${CONTAINER}:/tmp/ruler_raw.csv" 2>/dev/null
        docker exec ${CONTAINER} bash -c "cat /tmp/ruler_raw.csv >> ${CSV_SUMMARY}" 2>/dev/null
    fi

    echo ""
    echo "================================================================"
    echo "  RULER Long-Context Accuracy Summary"
    echo "================================================================"
    docker exec ${CONTAINER} cat "${CSV_SUMMARY}"

    # Print per-model comparison table
    echo ""
    echo "--- Per-model BF16 vs TQ accuracy comparison ---"
    docker exec -i ${CONTAINER} python3 - <<'PY' 2>/dev/null || true
import csv, sys
from collections import defaultdict

data = defaultdict(dict)
with open("${CSV_SUMMARY}") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['model'], row['context_length'], row['task'])
        config = row['config']
        score = row['score']
        data[key][config] = score

print(f"{'Model':<14} {'Ctx':>6} {'Task':<20} {'BF16':>8} {'TQ_4bit':>8} {'Delta':>8}")
print("-" * 70)
for (model, ctx, task), scores in sorted(data.items()):
    bf16 = scores.get('bf16', 'N/A')
    tq = scores.get('turboquant_4bit_nc', 'N/A')
    delta = ''
    try:
        if bf16 not in ('N/A', 'OOM') and tq not in ('N/A', 'OOM'):
            delta = f"{float(tq) - float(bf16):+.4f}"
        elif bf16 == 'OOM' and tq not in ('N/A', 'OOM'):
            delta = "BF16 OOM"
    except ValueError:
        pass
    print(f"{model:<14} {ctx:>6} {task:<20} {bf16:>8} {tq:>8} {delta:>8}")
PY
}

# ===========================================================================
echo "============================================"
echo "  RULER Long-Context Accuracy Benchmark"
echo "  Models:     ${MODEL_SHORTS[*]}"
echo "  Configs:    ${CONFIGS[*]}"
echo "  Context:    ${CONTEXT_LENGTHS[*]}"
echo "  Tasks:      ${RULER_TASKS}"
echo "  Timestamp:  ${TIMESTAMP}"
echo "  BF16 included as canary (expect OOM at long context)"
echo "============================================"

docker exec ${CONTAINER} mkdir -p ${RESULT_DIR}
: > "/tmp/ruler_raw_${TIMESTAMP}.csv"

# Run models sequentially — lm_eval is CPU-heavy and each eval takes a while
FAIL=0
for ms in "${MODEL_SHORTS[@]}"; do
    run_model_suite "${ms}" || FAIL=1
done

emit_summary
exit ${FAIL}
