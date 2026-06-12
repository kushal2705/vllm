#!/bin/bash
# bench_fp8_kv_perf.sh — Compare bf16 KV cache vs FP8 KV cache across 6 serving scenarios.
#
# Matches the benchmarks/sliding_window suite: same server args, same 6 scenarios
# (short_decode, long_prefill, mixed, high_load, very_long_prefill, decode_heavy),
# same bench parameters (openai backend, /v1/completions, max-concurrency, percentiles).
#
# All 10 models run sequentially (one at a time) to avoid GPU contention between
# benchmark runs. All models use --quantization fp8 (FP8 weight quantization).
# The benchmark compares --kv-cache-dtype bf16 (default) vs --kv-cache-dtype fp8;
# FP8 KV cache halves KV memory vs bf16, enabling ~2x more concurrent sequences.
#
# Usage:
#   bash bench_fp8_kv_perf.sh               # run all 10 models sequentially
#   bash bench_fp8_kv_perf.sh --all         # same as above
#   bash bench_fp8_kv_perf.sh llama31 qwen3 # run only these two models
#
# Supported model shorthands:
#   llama31    → meta-llama/Llama-3.1-8B-Instruct
#   deepseekr1 → deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#   gemma3     → google/gemma-3-1b-it
#   gemma4     → google/gemma-4-E4B-it
#   qwen3      → Qwen/Qwen3-8B
#   qwen25     → Qwen/Qwen2.5-14B-Instruct
#   mistral    → mistralai/Mistral-Small-24B-Instruct-2501
#   llama33_70b  → meta-llama/Llama-3.3-70B-Instruct       (TP=4)
#   qwen25_72b   → Qwen/Qwen2.5-72B-Instruct               (TP=4)
#   llama31_fp8  → nvidia/Llama-3.1-8B-Instruct-FP8       (TP=1, BF16 KV only)
#
# Runs directly inside the vllm container (no docker exec needed).
# Each config: start server → warm up → run 6 scenarios → stop server.
# Results: /tmp/fp8kv_perf/   Logs: /tmp/fp8kv_perf/logs/
# Reports per-scenario perf tables + KV cache memory comparison (fp8 vs bf16).
set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export VLLM_TARGET_DEVICE=xpu
export VLLM_MLA_DISABLE=1
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT=900
export VLLM_NO_USAGE_STATS=1
export HF_HOME=/tmp   # container /tmp is bind-mounted to ~/LLM on host

BASE_PORT=8192
MAX_MODEL_LEN=8192

# 6 scenarios from benchmarks/sliding_window/scenarios.tsv
# Format: name:input_len:output_len:num_prompts:concurrency
SCENARIOS=(
    "short_decode:128:512:200:32"
    "long_prefill:4096:128:200:32"
    "mixed:512:512:200:32"
    "high_load:512:128:500:64"
    "very_long_prefill:7168:64:200:16"
    "decode_heavy:64:1024:200:32"
)

# Configs: bf16 baseline vs FP8 KV cache
# fp8 uses --kv-cache-dtype fp8; bf16 omits the flag (default bf16 KV)
ALL_CONFIGS=("bf16" "fp8")

RESULT_DIR="/tmp/fp8kv_perf"
LOG_DIR="/tmp/fp8kv_perf/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
resolve_model() {
    case "$1" in
        llama31)        echo "meta-llama/Llama-3.1-8B-Instruct" ;;
        llama31_fp8)    echo "nvidia/Llama-3.1-8B-Instruct-FP8" ;;
        deepseekr1)     echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
        gemma3)         echo "google/gemma-3-1b-it" ;;
        gemma4)         echo "google/gemma-4-E4B-it" ;;
        qwen3)          echo "Qwen/Qwen3-8B" ;;
        qwen25)         echo "Qwen/Qwen2.5-14B-Instruct" ;;
        mistral)        echo "mistralai/Mistral-Small-24B-Instruct-2501" ;;
        llama33_70b)    echo "meta-llama/Llama-3.3-70B-Instruct" ;;
        qwen25_72b)     echo "Qwen/Qwen2.5-72B-Instruct" ;;
        deepseekr1_70b) echo "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" ;;
        *)              echo "UNKNOWN"; return 1 ;;
    esac
}

resolve_extra_args() {
    case "$1" in
        deepseekr1|deepseekr1_70b) echo "--trust-remote-code --quantization fp8" ;;
        gemma4)         echo "--trust-remote-code --attention-backend TRITON_ATTN --quantization fp8" ;;
        # nvidia/Llama-3.1-8B-Instruct-FP8: weights already statically FP8 — no flags needed
        llama31_fp8)    echo "" ;;
        *)              echo "--quantization fp8" ;;
    esac
}

# Returns space-separated list of TP values to benchmark for a model.
# 70B/72B models run TP=4 only (require 4 cards to fit in VRAM).
# Large models (gemma4, qwen25) run both TP=1 and TP=2 to compare.
# mistral runs TP=2 only (24B model, uses --quantization fp8).
# Small models (llama31, deepseekr1, gemma3, qwen3) run TP=1 only.
resolve_tp_list() {
    case "$1" in
        llama33_70b|qwen25_72b|deepseekr1_70b) echo "4" ;;
        gemma4|qwen25) echo "1 2" ;;
        mistral)       echo "2" ;;
        llama31_fp8)   echo "1" ;;
        *)             echo "1" ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse model arguments — each gets a card and port
# ---------------------------------------------------------------------------
ALL_MODELS=(llama31 deepseekr1 gemma3 qwen3 gemma4 qwen25 mistral llama33_70b qwen25_72b deepseekr1_70b llama31_fp8)

if [[ $# -eq 0 || "$1" == "--all" ]]; then
    set -- "${ALL_MODELS[@]}"
fi
MODEL_SHORTS=("$@")
for ms in "${MODEL_SHORTS[@]}"; do
    if ! resolve_model "${ms}" > /dev/null 2>&1; then
        echo "Unknown model: ${ms}. Choose from: llama31, llama31_fp8, deepseekr1, gemma3, gemma4, qwen3, qwen25, mistral, llama33_70b, qwen25_72b, deepseekr1_70b"; exit 1
    fi
done

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
start_server() {
    local model_short="$1" config="$2" model="$3" extra_args="$4" card="$5" port="$6" tp="$7"
    local server_log="${LOG_DIR}/server_${model_short}_${config}_tp${tp}.log"

    echo "[${model_short}] Starting server: config=${config} tp=${tp} card=${card} port=${port}"

    local kv_arg=""
    if [[ "${config}" != "bf16" ]]; then
        kv_arg="--kv-cache-dtype ${config}"
    fi

    # Build affinity mask: TP=1→single card, TP=2→two cards, TP=4→four cards
    # llama33_70b, qwen25_72b, deepseekr1_70b: skip ZE_AFFINITY_MASK so vLLM auto-detects all available cards
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
    if [[ "${model_short}" == "llama33_70b" || "${model_short}" == "qwen25_72b" || \
          "${model_short}" == "deepseekr1_70b" || "${model_short}" == "llama31_fp8" ]]; then
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
        --max-num-seq 64 \
        --block-size 64 \
        --no-enable-log-requests \
        --no-enable-prefix-caching \
        ${kv_arg} \
        ${extra_args} \
        > ${server_log} 2>&1"

    # Launch in a new session so its process group is isolated from this script.
    # setsid ensures kill -- -pgid won't propagate back to the script itself.
    setsid bash -c "${serve_cmd}" &
    local server_pid=$!
    echo "${server_pid}" > "${LOG_DIR}/pid_${model_short}.txt"

    echo "[${model_short}]     Waiting for server..."
    local attempts=0
    while ! curl -sf http://localhost:${port}/health > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ ${attempts} -gt 900 ]]; then
            echo "[${model_short}]     ERROR: Server failed to start after 900s"
            tail -30 "${server_log}" 2>/dev/null || true
            stop_server "${model_short}" "${port}"
            return 1
        fi
        sleep 1
    done
    echo "[${model_short}]     Server ready (${attempts}s)"

    # Warmup: 3 requests at concurrency 4
    echo "[${model_short}]     Warming up..."
    vllm bench serve \
        --model "${model}" \
        --backend openai \
        --endpoint /v1/completions \
        --port "${port}" \
        --dataset-name random \
        --random-input-len 64 --random-output-len 32 \
        --num-prompts 3 \
        --max-concurrency 4 \
        --ignore-eos \
        --disable-tqdm \
        > /dev/null 2>&1 || true
}

stop_server() {
    local model_short="$1" port="$2"
    local pid_file="${LOG_DIR}/pid_${model_short}.txt"
    echo "[${model_short}]     Stopping server on port ${port}..."
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        local pgid
        pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ') || true
        if [[ -n "${pgid}" && "${pgid}" != "0" ]]; then
            kill -- -"${pgid}" 2>/dev/null || true
        else
            kill "${pid}" 2>/dev/null || true
        fi
        rm -f "${pid_file}"
    fi
    # Fallback: catch any stragglers on this port
    pkill -f "vllm serve.*--port ${port}" 2>/dev/null || true
    sleep 3
}

# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------
run_scenario() {
    local model_short="$1" config="$2" scenario_spec="$3" model="$4" port="$5" tp="$6"
    IFS=':' read -r name in_len out_len n_prompts conc <<< "${scenario_spec}"
    local tag="${model_short}_${config}_tp${tp}__${name}__${TIMESTAMP}"

    local total=$((in_len + out_len))
    if [[ ${total} -gt ${MAX_MODEL_LEN} ]]; then
        echo "[${model_short}]     [${name}] SKIP: in+out=${total} > ${MAX_MODEL_LEN}"
        return 0
    fi

    echo "[${model_short}]     [${name}] in=${in_len} out=${out_len} n=${n_prompts} conc=${conc}"

    vllm bench serve \
        --model "${model}" \
        --backend openai \
        --endpoint /v1/completions \
        --port "${port}" \
        --dataset-name random \
        --random-input-len "${in_len}" \
        --random-output-len "${out_len}" \
        --num-prompts "${n_prompts}" \
        --max-concurrency "${conc}" \
        --ignore-eos \
        --save-result \
        --result-dir ${RESULT_DIR} \
        --result-filename "${tag}.json" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --metadata \
            model="${model}" \
            config="${config}" \
            scenario="${name}" \
            input_len="${in_len}" \
            output_len="${out_len}" \
            num_prompts="${n_prompts}" \
            concurrency="${conc}" \
        > "${LOG_DIR}/scenario_${tag}.log" 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[${model_short}]     [${name}] FAILED (exit ${rc})"
    else
        echo "[${model_short}]     [${name}] done"
    fi
}

# ---------------------------------------------------------------------------
# Extract KV cache size from server log
# ---------------------------------------------------------------------------
extract_kv_cache_tokens() {
    local model_short="$1" config="$2" tp="$3"
    grep -oP 'GPU KV cache size: \K[0-9,]+' "${LOG_DIR}/server_${model_short}_${config}_tp${tp}.log" 2>/dev/null \
        | tr -d ',' | tail -1 \
        || echo "N/A"
}

# ---------------------------------------------------------------------------
# run_model_suite — all configs × scenarios for one model (runs in background)
# ---------------------------------------------------------------------------
run_model_suite() {
    local model_short="$1" card="$2" port="$3"
    local model extra_args tp_list
    model=$(resolve_model "${model_short}")
    extra_args=$(resolve_extra_args "${model_short}")
    tp_list=$(resolve_tp_list "${model_short}")

    echo ""
    echo "[${model_short}] ========================================"
    echo "[${model_short}]  Model: ${model}"
    echo "[${model_short}]  Card: ${card}  Port: ${port}"
    echo "[${model_short}]  TP configs: ${tp_list}"
    echo "[${model_short}] ========================================"

    for tp in ${tp_list}; do
        for config in "${ALL_CONFIGS[@]}"; do
            # llama31_fp8 has pre-quantized FP8 weights — FP8 KV is redundant, run BF16 KV only
            if [[ "${model_short}" == "llama31_fp8" && "${config}" == "fp8" ]]; then
                echo "[${model_short}] Skipping fp8 KV (weights already FP8)"
                continue
            fi
            local run_tag="${config}_tp${tp}"
            echo ""
            echo "[${model_short}] --- Config: ${run_tag} ---"

            if start_server "${model_short}" "${config}" "${model}" "${extra_args}" "${card}" "${port}" "${tp}"; then
                local kv_tokens
                kv_tokens=$(extract_kv_cache_tokens "${model_short}" "${config}" "${tp}")
                echo "${kv_tokens}" > "${LOG_DIR}/kv_${model_short}_${config}_tp${tp}.txt"
                echo "[${model_short}]     KV cache tokens: ${kv_tokens}"

                for scenario in "${SCENARIOS[@]}"; do
                    run_scenario "${model_short}" "${config}" "${scenario}" "${model}" "${port}" "${tp}" || true
                done

                stop_server "${model_short}" "${port}"
            else
                echo "[${model_short}]     FAILED: ${run_tag} server did not start"
                echo "FAIL" > "${LOG_DIR}/kv_${model_short}_${config}_tp${tp}.txt"
                stop_server "${model_short}" "${port}"
            fi
        done
    done
}

# ---------------------------------------------------------------------------
# print_model_summary — KV compression + per-scenario perf tables
# ---------------------------------------------------------------------------
print_model_summary() {
    local model_short="$1"
    local model tp_list
    model=$(resolve_model "${model_short}")
    tp_list=$(resolve_tp_list "${model_short}")

    echo ""
    echo "================================================================"
    echo "  SUMMARY: ${model} (${model_short})"
    echo "================================================================"

    for tp in ${tp_list}; do
        echo ""
        echo "  ── TP=${tp} ──────────────────────────────────────────────────────"

        # KV Cache token count
        echo ""
        echo "--- KV Cache Tokens (TP=${tp}) ---"
        printf "  %-8s %14s %12s\n" "Config" "KV Cache Toks" "vs bf16"
        printf "  %-8s %14s %12s\n" "--------" "--------------" "------------"

        local bf16_tokens
        bf16_tokens=$(cat "${LOG_DIR}/kv_${model_short}_bf16_tp${tp}.txt" 2>/dev/null || echo "N/A")

        for config in "${ALL_CONFIGS[@]}"; do
            local tokens
            tokens=$(cat "${LOG_DIR}/kv_${model_short}_${config}_tp${tp}.txt" 2>/dev/null || echo "N/A")
            if [[ "${tokens}" == "FAIL" || "${tokens}" == "N/A" ]]; then
                printf "  %-8s %14s %12s\n" "${config}" "${tokens}" "-"
            elif [[ "${bf16_tokens}" != "N/A" && "${bf16_tokens}" != "FAIL" && "${bf16_tokens}" != "0" ]]; then
                local ratio
                ratio=$(python3 -c "print(f'{int(\"${tokens}\")/int(\"${bf16_tokens}\"):.2f}x')")
                printf "  %-8s %14s %12s\n" "${config}" "${tokens}" "${ratio}"
            else
                printf "  %-8s %14s %12s\n" "${config}" "${tokens}" "-"
            fi
        done

        # Per-scenario performance tables
        for scenario_spec in "${SCENARIOS[@]}"; do
            IFS=':' read -r name in_len out_len n_prompts conc <<< "${scenario_spec}"
            local total=$((in_len + out_len))
            if [[ ${total} -gt ${MAX_MODEL_LEN} ]]; then
                continue
            fi

            echo ""
            echo "--- ${name}  (ISL=${in_len}, OSL=${out_len}, N=${n_prompts}, C=${conc}, TP=${tp}) ---"
            printf "  %-8s %8s %10s %10s %10s %10s %10s %10s %10s\n" \
                "Config" "Req/s" "OutTok/s" "TTFT" "TPOT" "ITL" "p90 TTFT" "p90 TPOT" "p99 TTFT"
            printf "  %-8s %8s %10s %10s %10s %10s %10s %10s %10s\n" \
                "--------" "--------" "----------" "----------" "----------" "----------" "----------" "----------" "----------"

            for config in "${ALL_CONFIGS[@]}"; do
                local tag="${model_short}_${config}_tp${tp}__${name}__${TIMESTAMP}"
                local json_file="${RESULT_DIR}/${tag}.json"

                local metrics
                metrics=$(python3 -c "
import json, sys
try:
    with open('${json_file}') as f:
        d = json.load(f)
    req = d.get('request_throughput', 0)
    out = d.get('output_throughput', 0)
    ttft = d.get('mean_ttft_ms', 0)
    tpot = d.get('mean_tpot_ms', 0)
    itl = d.get('mean_itl_ms', 0)
    p90_ttft = d.get('p90_ttft_ms', 0)
    p90_tpot = d.get('p90_tpot_ms', 0)
    p99_ttft = d.get('p99_ttft_ms', 0)
    print(f'{req:.1f}|{out:.0f}|{ttft:.0f}|{tpot:.1f}|{itl:.1f}|{p90_ttft:.0f}|{p90_tpot:.1f}|{p99_ttft:.0f}')
except Exception:
    print('-|-|-|-|-|-|-|-')
" 2>/dev/null || echo "-|-|-|-|-|-|-|-")

                IFS='|' read -r req_s out_s ttft_s tpot_s itl_s p90ttft_s p90tpot_s p99ttft_s <<< "${metrics}"
                printf "  %-8s %8s %10s %10s %10s %10s %10s %10s %10s\n" \
                    "${config}" "${req_s}" "${out_s}" "${ttft_s}" "${tpot_s}" "${itl_s}" "${p90ttft_s}" "${p90tpot_s}" "${p99ttft_s}"
            done
        done
    done
}

# ===========================================================================
# Main
# ===========================================================================
echo "============================================"
echo "  FP8 KV Cache Performance Benchmark"
echo "  Models: ${MODEL_SHORTS[*]}"
echo "  Scenarios: ${#SCENARIOS[@]} (sliding_window suite)"
echo "  Configs: bf16 vs fp8 KV cache"
echo "  max_model_len: ${MAX_MODEL_LEN}"
echo "  timestamp: ${TIMESTAMP}"
echo "============================================"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# Always sequential — parallel execution skews benchmark results (GPU contention)
echo "  Mode: sequential (${#MODEL_SHORTS[@]} model(s), one at a time)"
FAIL=0
for ms in "${MODEL_SHORTS[@]}"; do
    run_model_suite "${ms}" "0" "${BASE_PORT}" || FAIL=1
    print_model_summary "${ms}"
done

echo ""
echo "Theoretical KV compression: fp8=2.0x vs bf16 (8-bit vs 16-bit KV cache)"
echo "Logs: ${LOG_DIR}/"
echo "Results: ls ${RESULT_DIR}/"

exit ${FAIL}
