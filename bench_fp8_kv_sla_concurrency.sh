#!/bin/bash
# bench_fp8_kv_sla_concurrency.sh — SLA-bound max concurrency for FP8 KV cache.
#
# Serving SLA: TTFT(p99) ≤ 5000 ms AND TPOT(p99) ≤ 200 ms.
# For each (model, TP, config), find the highest concurrency that meets the SLA.
# Report concurrency multiplier vs BF16 per model.
#
# Configs: bf16 (baseline) vs fp8 KV cache.
#
# Per-model TP:
#   TP=1: llama31, deepseekr1, gemma3, qwen3
#   TP=1+2: gemma4, qwen25
#   TP=2 only: mistral
#   TP=4 only: llama33_70b, qwen25_72b, deepseekr1_70b
#
# Scheduling:
#   TP=1 models run in PARALLEL on separate cards (one per card).
#   TP=2, TP=4 models run SEQUENTIALLY (need multiple cards).
#   Models with multiple TP configs (gemma4, qwen25) run each TP sequentially.
#
# Strategy: linear sweep with early-exit after 2 consecutive SLA failures.
#
# Output:
#   /tmp/fp8kv_sla/sla_sweep_<TS>.csv     — every measured point (host: ~/LLM/fp8kv_sla/)
#   /tmp/fp8kv_sla/sla_summary_<TS>.csv   — max conc passing + ratio
set -euo pipefail

# ── Environment (required for Intel XPU backend) ────────────────────────────
export VLLM_TARGET_DEVICE=xpu
export VLLM_MLA_DISABLE=1
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT=900
export VLLM_NO_USAGE_STATS=1
export HF_HOME=/tmp   # container /tmp is bind-mounted to ~/LLM on host

BASE_PORT=8220
MAX_MODEL_LEN=4096
INPUT_LEN=1024
OUTPUT_LEN=512

SLA_TTFT_MS=5000
SLA_TPOT_MS=200

CONCURRENCIES=(1 2 4 8 16 32 64 128 256)

RESULT_DIR="/tmp/fp8kv_sla"
LOG_DIR="${RESULT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV_SWEEP="${RESULT_DIR}/sla_sweep_${TIMESTAMP}.csv"
CSV_SUMMARY="${RESULT_DIR}/sla_summary_${TIMESTAMP}.csv"

ALL_MODELS=(llama31 deepseekr1 gemma3 qwen3 gemma4 qwen25 mistral llama33_70b qwen25_72b deepseekr1_70b)

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
        *)              echo "UNKNOWN"; return 1 ;;
    esac
}

# Returns space-separated list of TP values for this model.
resolve_tp_list() {
    case "$1" in
        llama33_70b|qwen25_72b|deepseekr1_70b) echo "4" ;;
        gemma4|qwen25) echo "1 2" ;;
        mistral)       echo "2" ;;
        *)             echo "1" ;;
    esac
}

# Extra vllm serve flags per model.
resolve_extra_args() {
    case "$1" in
        deepseekr1|deepseekr1_70b) echo "--trust-remote-code --quantization fp8" ;;
        gemma4)         echo "--trust-remote-code --attention-backend TRITON_ATTN --quantization fp8" ;;
        *)              echo "--quantization fp8" ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse arguments:
#   --configs <c1,c2>  override which configs to run (default: bf16,fp8)
#   remaining args     model shorthands (default: all models)
CONFIGS=("bf16" "fp8")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs)
            IFS=',' read -ra CONFIGS <<< "$2"
            shift 2
            ;;
        *)
            break
            ;;
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

# ---------------------------------------------------------------------------
start_server() {
    local model_short="$1" config="$2" model="$3" extra_args="$4" card="$5" port="$6" tp="$7"
    local server_log="${LOG_DIR}/vllm_sla_server_${model_short}_${config}_tp${tp}.log"

    echo "[${model_short}] Starting: config=${config} tp=${tp} card=${card} port=${port}"

    local kv_arg=""
    [[ "${config}" != "bf16" ]] && kv_arg="--kv-cache-dtype ${config}"

    # 70B models: skip ZE_AFFINITY_MASK and --dtype, let vLLM auto-detect cards.
    local ze_prefix=""
    if [[ "${model_short}" != "llama33_70b" && "${model_short}" != "qwen25_72b" && "${model_short}" != "deepseekr1_70b" ]]; then
        local affinity_mask="${card}"
        if [[ "${tp}" -eq 2 ]]; then
            affinity_mask="${card},$(( card + 1 ))"
        elif [[ "${tp}" -eq 4 ]]; then
            affinity_mask="${card},$(( card + 1 )),$(( card + 2 )),$(( card + 3 ))"
        fi
        ze_prefix="ZE_AFFINITY_MASK=${affinity_mask}"
    fi

    local dtype_arg="--dtype bfloat16"
    if [[ "${model_short}" == "llama33_70b" || "${model_short}" == "qwen25_72b" || "${model_short}" == "deepseekr1_70b" ]]; then
        dtype_arg=""
    fi

    local serve_cmd="${ze_prefix} VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
        vllm serve ${model} \
        --port ${port} \
        ${dtype_arg} \
        --tensor-parallel-size ${tp} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization 0.92 \
        --enforce-eager \
        --max-num-batched-tokens 8192 \
        --max-num-seq 256 \
        --block-size 64 \
        --no-enable-log-requests \
        --no-enable-prefix-caching \
        ${kv_arg} \
        ${extra_args} \
        > ${server_log} 2>&1"

    setsid bash -c "${serve_cmd}" &
    local server_pid=$!
    echo "${server_pid}" > "${LOG_DIR}/vllm_sla_pid_${model_short}.txt"

    echo "[${model_short}]   Waiting for server (up to 900s)..."
    local attempts=0
    while ! curl -sf http://localhost:${port}/health > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ ${attempts} -gt 900 ]]; then
            echo "[${model_short}]   ERROR: server failed to start after 900s"
            tail -30 "${server_log}" 2>/dev/null || true
            stop_server "${model_short}" "${port}"
            return 1
        fi
        sleep 1
    done
    echo "[${model_short}]   Server ready in ${attempts}s"

    # Warmup
    vllm bench serve \
        --model "${model}" --backend openai --endpoint /v1/completions \
        --port "${port}" --dataset-name random \
        --random-input-len 64 --random-output-len 32 \
        --num-prompts 4 --max-concurrency 4 \
        --ignore-eos --disable-tqdm > /dev/null 2>&1 || true
}

stop_server() {
    local model_short="$1" port="$2"
    echo "[${model_short}]   Stopping server on port ${port}..."
    local pid_file="${LOG_DIR}/vllm_sla_pid_${model_short}.txt"
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
    for i in $(seq 1 30); do
        ss -tln 2>/dev/null | grep -q ":${port} " || break
        sleep 1
    done
    sleep 3
}

# ---------------------------------------------------------------------------
run_point() {
    local model_short="$1" config="$2" tp="$3" conc="$4" model="$5" port="$6"
    local n_prompts=$(( conc * 4 ))
    [[ ${n_prompts} -lt 20 ]] && n_prompts=20
    [[ ${conc} -ge 8 && ${n_prompts} -lt 64 ]] && n_prompts=64
    [[ ${n_prompts} -gt 512 ]] && n_prompts=512

    local tag="sla_${model_short}_${config}_tp${tp}_c${conc}_${TIMESTAMP}"
    local bench_log="${LOG_DIR}/bench_${tag}.log"

    echo "[${model_short}]     Running ${n_prompts} prompts at conc=${conc} (ISL=${INPUT_LEN} OSL=${OUTPUT_LEN})..." >&2
    local t_start
    t_start=$(date +%s)

    vllm bench serve \
        --model "${model}" --backend openai --endpoint /v1/completions \
        --port "${port}" --dataset-name random \
        --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" \
        --num-prompts "${n_prompts}" \
        --max-concurrency "${conc}" \
        --ignore-eos --disable-tqdm \
        --save-result --result-dir "${RESULT_DIR}" \
        --result-filename "${tag}.json" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --metadata model="${model}" config="${config}" tp="${tp}" concurrency="${conc}" \
        > "${bench_log}" 2>&1 || true

    local t_end elapsed
    t_end=$(date +%s)
    elapsed=$(( t_end - t_start ))
    echo "[${model_short}]     bench done in ${elapsed}s — parsing results..." >&2

    python3 - <<PY 2>/dev/null || echo "FAIL|FAIL|FAIL"
import json, os, sys
p = "${RESULT_DIR}/${tag}.json"
if not os.path.exists(p):
    print("FAIL|FAIL|FAIL"); sys.exit(0)
with open(p) as f:
    d = json.load(f)
tps = d.get('output_throughput', 0) or 0
if tps <= 0 or d.get('completed', 0) == 0:
    print("FAIL|FAIL|FAIL"); sys.exit(0)
print(f"{d.get('p99_ttft_ms',0):.2f}|{d.get('p99_tpot_ms',0):.2f}|{tps:.2f}")
PY
}

# ---------------------------------------------------------------------------
sweep_config() {
    local model_short="$1" config="$2" tp="$3" model="$4" extra_args="$5" card="$6" port="$7"

    if ! start_server "${model_short}" "${config}" "${model}" "${extra_args}" "${card}" "${port}" "${tp}"; then
        stop_server "${model_short}" "${port}"
        echo "0" > "${LOG_DIR}/sla_max_${model_short}_${config}_tp${tp}.txt"
        return
    fi

    local max_pass=0
    local consec_fail=0
    for conc in "${CONCURRENCIES[@]}"; do
        echo "[${model_short}]   ${config}_tp${tp} conc=${conc}"
        local res
        res=$(run_point "${model_short}" "${config}" "${tp}" "${conc}" "${model}" "${port}")
        IFS='|' read -r p99_ttft p99_tpot out_tps <<< "${res}"

        local pass="0"
        if [[ "${p99_ttft}" != "FAIL" && -n "${p99_ttft}" ]]; then
            pass=$(python3 -c "
ttft=${p99_ttft}; tpot=${p99_tpot}; tps=${out_tps}
if tps <= 0 or (ttft == 0 and tpot == 0):
    print(0)
else:
    print(1 if (ttft <= ${SLA_TTFT_MS} and tpot <= ${SLA_TPOT_MS}) else 0)
")
        fi

        echo "${model_short},${config},${tp},${conc},${p99_ttft},${p99_tpot},${out_tps},${pass}" \
            >> "${LOG_DIR}/sla_sweep_${TIMESTAMP}.csv"

        echo "[${model_short}]     p99_ttft=${p99_ttft}ms p99_tpot=${p99_tpot}ms out_tps=${out_tps} pass=${pass}"

        if [[ "${pass}" == "1" ]]; then
            max_pass="${conc}"
            consec_fail=0
        else
            consec_fail=$((consec_fail + 1))
            if [[ ${consec_fail} -ge 2 ]]; then
                echo "[${model_short}]   2 consecutive fails — stopping sweep"
                break
            fi
        fi
    done

    echo "${max_pass}" > "${LOG_DIR}/sla_max_${model_short}_${config}_tp${tp}.txt"
    stop_server "${model_short}" "${port}"
}

# ---------------------------------------------------------------------------
run_model_suite() {
    local model_short="$1" card="$2" port="$3" tp_override="${4:-}"
    local model extra_args tp_list
    model=$(resolve_model "${model_short}")
    extra_args=$(resolve_extra_args "${model_short}")
    tp_list=$(resolve_tp_list "${model_short}")
    # If a specific TP is requested, restrict to that TP only
    [[ -n "${tp_override}" ]] && tp_list="${tp_override}"

    echo ""
    echo "[${model_short}] ========================================"
    echo "[${model_short}]  Model: ${model}"
    echo "[${model_short}]  Card: ${card}  Port: ${port}"
    echo "[${model_short}]  TP configs: ${tp_list}"
    echo "[${model_short}] ========================================"

    for tp in ${tp_list}; do
        for config in "${CONFIGS[@]}"; do
            echo ""
            echo "[${model_short}] --- Config: ${config}_tp${tp} ---"
            sweep_config "${model_short}" "${config}" "${tp}" "${model}" "${extra_args}" "${card}" "${port}"
        done
    done
}

# ---------------------------------------------------------------------------
emit_summary() {
    mkdir -p "${RESULT_DIR}"

    (echo "model,config,tp,concurrency,p99_ttft_ms,p99_tpot_ms,output_tps,sla_pass"; \
        cat "${LOG_DIR}/sla_sweep_${TIMESTAMP}.csv" 2>/dev/null) > "${CSV_SWEEP}"

    echo "model,config,tp,max_conc_passing_sla,ratio_vs_bf16" > "${CSV_SUMMARY}"
    for ms in "${MODEL_SHORTS[@]}"; do
        local tp_list
        tp_list=$(resolve_tp_list "${ms}")
        for tp in ${tp_list}; do
            local bf16_max
            bf16_max=$(cat "${LOG_DIR}/sla_max_${ms}_bf16_tp${tp}.txt" 2>/dev/null || echo "0")
            for config in "${CONFIGS[@]}"; do
                local m
                m=$(cat "${LOG_DIR}/sla_max_${ms}_${config}_tp${tp}.txt" 2>/dev/null || echo "0")
                local ratio="-"
                if [[ "${bf16_max}" != "0" && "${bf16_max}" != "FAIL" ]]; then
                    ratio=$(python3 -c "print(f'{int(\"${m}\")/int(\"${bf16_max}\"):.2f}x')" 2>/dev/null || echo "-")
                fi
                echo "${ms},${config},${tp},${m},${ratio}" >> "${CSV_SUMMARY}"
            done
        done
    done

    echo ""
    echo "================================================================"
    echo "  SLA: TTFT(p99) ≤ ${SLA_TTFT_MS}ms  AND  TPOT(p99) ≤ ${SLA_TPOT_MS}ms"
    echo "================================================================"
    echo ""
    echo "Per-point sweep: ${CSV_SWEEP}"
    cat "${CSV_SWEEP}"
    echo ""
    echo "Summary (max passing concurrency): ${CSV_SUMMARY}"
    cat "${CSV_SUMMARY}"
}

# ===========================================================================
echo "============================================"
echo "  FP8 KV Cache SLA-bound Max Concurrency"
echo "  Models:   ${MODEL_SHORTS[*]}"
echo "  Configs:  ${CONFIGS[*]}"
echo "  SLA:      TTFT(p99)≤${SLA_TTFT_MS}ms, TPOT(p99)≤${SLA_TPOT_MS}ms"
echo "  Conc:     ${CONCURRENCIES[*]}"
echo "  ISL/OSL:  ${INPUT_LEN}/${OUTPUT_LEN}  max_model_len=${MAX_MODEL_LEN}"
echo "  Timestamp: ${TIMESTAMP}"
echo "============================================"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
: > "${LOG_DIR}/sla_sweep_${TIMESTAMP}.csv"

# Kill any leftover vllm processes from previous runs before starting
echo "Checking for leftover vllm processes..."
if pgrep -f "vllm serve" > /dev/null 2>&1 || pgrep -f "EngineCore" > /dev/null 2>&1; then
    echo "  Found stale vllm processes — killing before starting benchmark..."
    pkill -TERM -f "vllm serve" 2>/dev/null || true
    sleep 3
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    pkill -KILL -f "EngineCore" 2>/dev/null || true
    pkill -KILL -f "from multiprocessing.spawn" 2>/dev/null || true
    sleep 3
    echo "  Done."
else
    echo "  None found. Proceeding."
fi

# Split models by max TP
TP1_MODELS=()
TP2_MODELS=()
TP4_MODELS=()
for ms in "${MODEL_SHORTS[@]}"; do
    tp_list=$(resolve_tp_list "${ms}")
    max_tp=$(echo "${tp_list}" | tr ' ' '\n' | sort -n | tail -1)
    if [[ "${max_tp}" == "1" ]]; then
        TP1_MODELS+=("${ms}")
    elif [[ "${max_tp}" == "2" ]]; then
        # Models with tp_list="1 2" (gemma4, qwen25) need TP=1 in Phase 1 too.
        # Add them to TP1 for their TP=1 sweep, TP2 for their TP=2 sweep.
        if echo "${tp_list}" | grep -qw '1'; then
            TP1_MODELS+=("${ms}")
        fi
        TP2_MODELS+=("${ms}")
    else
        TP4_MODELS+=("${ms}")
    fi
done

FAIL=0

# --- Phase 1: TP=1 models in parallel batches of 4 (one per card) ---
if [[ ${#TP1_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo ">>> Phase 1: TP=1 models in parallel batches (${TP1_MODELS[*]})"
    i=0
    PIDS=()
    for ms in "${TP1_MODELS[@]}"; do
        slot=$(( i % 4 ))
        card="${slot}"
        port=$((BASE_PORT + slot))
        run_model_suite "${ms}" "${card}" "${port}" "1" &
        PIDS+=($!)
        i=$((i + 1))
        # Every 4 launches, wait for the batch before starting more
        if (( i % 4 == 0 )); then
            for pid in "${PIDS[@]}"; do wait "${pid}" || FAIL=1; done
            PIDS=()
            sleep 5
        fi
    done
    # Wait for any remaining models in the last (possibly partial) batch
    for pid in "${PIDS[@]}"; do wait "${pid}" || FAIL=1; done
    echo ">>> Phase 1 complete"
fi

# --- Phase 2: TP=2 models in parallel pairs (cards 0,1 and cards 2,3) ---
if [[ ${#TP2_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo ">>> Phase 2: TP=2 models in parallel pairs (${TP2_MODELS[*]})"
    sleep 5
    PIDS=()
    i=0
    for ms in "${TP2_MODELS[@]}"; do
        # Slot 0 → cards 0,1  port BASE_PORT
        # Slot 1 → cards 2,3  port BASE_PORT+1
        slot=$(( i % 2 ))
        card=$(( slot * 2 ))
        port=$((BASE_PORT + slot))
        run_model_suite "${ms}" "${card}" "${port}" "2" &
        PIDS+=($!)
        i=$((i + 1))
        # After filling both slots (all 4 cards used), wait before launching more
        if [[ $(( i % 2 )) -eq 0 ]]; then
            for pid in "${PIDS[@]}"; do
                wait "${pid}" || FAIL=1
            done
            PIDS=()
            sleep 5
        fi
    done
    # wait for any remaining (odd number of TP=2 models)
    for pid in "${PIDS[@]}"; do
        wait "${pid}" || FAIL=1
    done
    echo ">>> Phase 2 complete"
fi

# --- Phase 3: TP=4 models sequentially (need all 4 cards) ---
if [[ ${#TP4_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo ">>> Phase 3: TP=4 models sequentially (${TP4_MODELS[*]})"
    sleep 5
    for ms in "${TP4_MODELS[@]}"; do
        run_model_suite "${ms}" "0" "${BASE_PORT}" "4" || FAIL=1
    done
    echo ">>> Phase 3 complete"
fi

emit_summary
exit ${FAIL}
