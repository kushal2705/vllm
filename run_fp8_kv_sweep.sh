#!/usr/bin/env bash
# =============================================================================
# run_fp8_kv_sweep.sh
# vLLM FP8 KV-cache sweep — BF16 KV vs FP8 KV, any model
# Hardware: Intel® Arc™ Pro B70 32 GiB GDDR6 (cards 0–3)
# vLLM: v0.20.1rc1.dev105+g3ca6ca210.d20260509
#
# Usage:
#   ./run_fp8_kv_sweep.sh                          # default: gemma4 qwen25 mistral (TP=2, 2 parallel)
#   ./run_fp8_kv_sweep.sh --all                    # all 10 models (parallel by TP group)
#   ./run_fp8_kv_sweep.sh llama31                  # single model by shortname
#   ./run_fp8_kv_sweep.sh llama31 qwen3 gemma3     # multiple models
#   ./run_fp8_kv_sweep.sh --kv fp8 gemma4          # FP8 KV only
#   ./run_fp8_kv_sweep.sh --kv bf16 llama31        # BF16 KV only
#   ./run_fp8_kv_sweep.sh --base-port 8010         # override base port (default 8000)
#
# Parallelism:
#   TP=1 models: up to 4 in parallel, one per card (ports BASE_PORT+0..3)
#   TP=2 models: up to 2 in parallel (cards 0,1 and 2,3; ports BASE_PORT+0,1)
#   TP=4 models: sequential (all 4 cards per model)
#
# Shortnames: llama31 deepseekr1 gemma3 qwen3 gemma4 qwen25 mistral
#             llama33_70b qwen25_72b deepseekr1_70b llama31_fp8
# =============================================================================
set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export VLLM_TARGET_DEVICE=xpu
export VLLM_MLA_DISABLE=1
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT=900
export VLLM_NO_USAGE_STATS=1
export HF_HOME=/tmp   # container /tmp is bind-mounted to ~/LLM on host; models cached at /tmp/hub/

# ── Defaults ──────────────────────────────────────────────────────────────────
LOG_DIR="/tmp/fp8_kv_sweep"
BASE_PORT=8000
READY_TIMEOUT=900
HEALTHCHECK_INTERVAL=5
KV_DTYPES=("bf16" "fp8")   # run both by default
REQUESTED_MODELS=()
ALL_MODELS=(llama31 deepseekr1 gemma3 qwen3 gemma4 qwen25 mistral llama33_70b qwen25_72b deepseekr1_70b llama31_fp8)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            REQUESTED_MODELS=("${ALL_MODELS[@]}"); shift ;;
        --kv)
            case "$2" in
                bf16)  KV_DTYPES=("bf16") ;;
                fp8)   KV_DTYPES=("fp8")  ;;
                both)  KV_DTYPES=("bf16" "fp8") ;;
                *)     echo "ERROR: --kv must be bf16, fp8, or both"; exit 1 ;;
            esac
            shift 2 ;;
        --base-port)
            BASE_PORT="$2"; shift 2 ;;
        --log-dir)
            LOG_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,22p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*)
            echo "ERROR: Unknown flag: $1"; exit 1 ;;
        *)
            REQUESTED_MODELS+=("$1"); shift ;;
    esac
done

# ── Model registry ────────────────────────────────────────────────────────────
resolve_hf_id() {
    case "$1" in
        llama31)        echo "meta-llama/Llama-3.1-8B-Instruct" ;;
        llama31_fp8)    echo "nvidia/Llama-3.1-8B-Instruct-FP8" ;;
        deepseekr1)     echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
        gemma3)         echo "google/gemma-3-1b-it" ;;
        qwen3)          echo "Qwen/Qwen3-8B" ;;
        gemma4)         echo "google/gemma-4-E4B-it" ;;
        qwen25)         echo "Qwen/Qwen2.5-14B-Instruct" ;;
        mistral)        echo "mistralai/Mistral-Small-24B-Instruct-2501" ;;
        llama33_70b)    echo "meta-llama/Llama-3.3-70B-Instruct" ;;
        qwen25_72b)     echo "Qwen/Qwen2.5-72B-Instruct" ;;
        deepseekr1_70b) echo "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" ;;
        *)              echo "$1" ;;   # pass through HF IDs directly
    esac
}

resolve_tp() {
    case "$1" in
        gemma4|qwen25|mistral)                  echo "2" ;;
        llama33_70b|qwen25_72b|deepseekr1_70b)  echo "4" ;;
        *)                                       echo "1" ;;
    esac
}

resolve_extra_args() {
    case "$1" in
        deepseekr1|deepseekr1_70b) echo "--trust-remote-code --quantization fp8" ;;
        gemma4)             echo "--trust-remote-code --attention-backend TRITON_ATTN --quantization fp8" ;;
        # nvidia/Llama-3.1-8B-Instruct-FP8: weights + KV cache already statically FP8
        # (hf_quant_config.json has kv_cache_quant_algo=FP8) — no flags needed
        llama31_fp8)        echo "" ;;
        *)                  echo "--quantization fp8" ;;
    esac
}

# ── Resolve model list ────────────────────────────────────────────────────────
# Default to original 3 TP=2 models if none specified
if [[ ${#REQUESTED_MODELS[@]} -eq 0 ]]; then
    REQUESTED_MODELS=(gemma4 qwen25 mistral)
fi

mkdir -p "${LOG_DIR}"

# ── Helper: stop a server by PID ─────────────────────────────────────────────
stop_server() {
    local pid="$1" label="${2:-server}"
    local pgid
    pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ') || true
    if [[ -n "${pgid}" && "${pgid}" != "0" ]]; then
        kill -- -"${pgid}" 2>/dev/null || true
    else
        kill "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
    sleep 2
    echo "[${label}] Server stopped."
}

# ── run_model_suite — BF16+FP8 KV sweep for one model (run in background) ────
run_model_suite() {
    local model_short="$1" card="$2" port="$3"
    local hf_id tp extra_args
    hf_id="$(resolve_hf_id "${model_short}")"
    tp="$(resolve_tp "${model_short}")"
    extra_args="$(resolve_extra_args "${model_short}")"

    echo ""
    echo "[${model_short}] ========================================"
    echo "[${model_short}]  Model: ${hf_id}  TP: ${tp}  Card: ${card}  Port: ${port}"
    echo "[${model_short}] ========================================"

    # ZE_AFFINITY_MASK — skip for 70B models (they use all 4 cards automatically)
    local ze_env=""
    if [[ "${model_short}" != "llama33_70b" && "${model_short}" != "qwen25_72b" && "${model_short}" != "deepseekr1_70b" ]]; then
        local affinity_mask="${card}"
        [[ "${tp}" -eq 2 ]] && affinity_mask="${card},$(( card + 1 ))"
        ze_env="ZE_AFFINITY_MASK=${affinity_mask}"
    fi

    # 70B models do not use --dtype bfloat16 (weight quant fp8 handles precision)
    local dtype_arg="--dtype bfloat16"
    if [[ "${model_short}" == "llama33_70b" || "${model_short}" == "qwen25_72b" || \
          "${model_short}" == "deepseekr1_70b" || "${model_short}" == "llama31_fp8" ]]; then
        dtype_arg=""
    fi

    for kv_dtype in "${KV_DTYPES[@]}"; do
        # For models with pre-quantized FP8 weights, FP8 KV is redundant — BF16 KV only
        if [[ "${model_short}" == "llama31_fp8" && "${kv_dtype}" == "fp8" ]]; then
            echo "[${model_short}] Skipping fp8 KV (weights already FP8)"
            continue
        fi
        local run_label="${model_short}__kv_${kv_dtype}_tp${tp}"
        local log_file="${LOG_DIR}/${run_label}.log"

        echo "[${model_short}] Starting server: kv=${kv_dtype}"

        local kv_arg=""
        [[ "${kv_dtype}" == "fp8" ]] && kv_arg="--kv-cache-dtype fp8"

        local serve_cmd="${ze_env} VLLM_NO_USAGE_STATS=1 vllm serve ${hf_id} \
            ${dtype_arg} \
            --tensor-parallel-size ${tp} \
            --max-model-len 4096 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 128 \
            --gpu-memory-utilization 0.9 \
            --enforce-eager \
            --no-enable-prefix-caching \
            --block-size 64 \
            --port ${port} \
            --host 0.0.0.0 \
            ${extra_args} \
            ${kv_arg} \
            > ${log_file} 2>&1"

        setsid bash -c "${serve_cmd}" &
        local server_pid=$!

        # Wait for healthy
        local elapsed=0 failed=0
        until curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
            sleep "${HEALTHCHECK_INTERVAL}"
            elapsed=$(( elapsed + HEALTHCHECK_INTERVAL ))
            if [[ ${elapsed} -ge ${READY_TIMEOUT} ]]; then
                echo "[${model_short}] ERROR: server failed to start (kv=${kv_dtype}) — skipping"
                stop_server "${server_pid}" "${model_short}"
                failed=1; break
            fi
        done
        [[ ${failed} -eq 1 ]] && continue

        echo "[${model_short}] Server ready (${elapsed}s) — kv=${kv_dtype}"

        # Capacity summary
        sleep 2
        grep -E "Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|Checkpoint size" \
            "${log_file}" | tee "${LOG_DIR}/${run_label}.summary.txt" || true

        # Smoke test
        echo "[${model_short}] Smoke test (kv=${kv_dtype}) ..."
        curl -sf "http://localhost:${port}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"${hf_id}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 8}" \
            | python3 -c "import sys,json; r=json.load(sys.stdin); print('[${model_short}] OK:', r['choices'][0]['message']['content'])" \
            || echo "[${model_short}] Smoke test failed (non-fatal)"

        stop_server "${server_pid}" "${model_short}"
        echo "[${model_short}] Done — kv=${kv_dtype}"
    done

    echo "[${model_short}] All configs complete."
}

# ── Parallel execution ────────────────────────────────────────────────────────
# Separate the requested models into TP groups
tp1_models=(); tp2_models=(); tp4_models=()
for m in "${REQUESTED_MODELS[@]}"; do
    case "$(resolve_tp "${m}")" in
        1) tp1_models+=("${m}") ;;
        2) tp2_models+=("${m}") ;;
        4) tp4_models+=("${m}") ;;
    esac
done

# TP=1: up to 4 models in parallel, one per card (ports BASE_PORT+0..3)
if [[ ${#tp1_models[@]} -gt 0 ]]; then
    echo "=== TP=1: ${#tp1_models[@]} model(s) — up to 4 parallel ==="
    batch_pids=(); slot=0
    for m in "${tp1_models[@]}"; do
        run_model_suite "${m}" "${slot}" "$(( BASE_PORT + slot ))" &
        batch_pids+=($!)
        slot=$(( slot + 1 ))
        if [[ ${#batch_pids[@]} -eq 4 ]]; then
            wait "${batch_pids[@]}"; batch_pids=(); slot=0
        fi
    done
    [[ ${#batch_pids[@]} -gt 0 ]] && wait "${batch_pids[@]}"
    echo "=== TP=1 complete ==="
fi

# TP=2: up to 2 models in parallel (slot 0 → cards 0,1 ; slot 1 → cards 2,3)
if [[ ${#tp2_models[@]} -gt 0 ]]; then
    echo "=== TP=2: ${#tp2_models[@]} model(s) — up to 2 parallel ==="
    batch_pids=(); slot=0
    for m in "${tp2_models[@]}"; do
        card=$(( slot * 2 ))
        run_model_suite "${m}" "${card}" "$(( BASE_PORT + slot ))" &
        batch_pids+=($!)
        slot=$(( slot + 1 ))
        if [[ ${#batch_pids[@]} -eq 2 ]]; then
            wait "${batch_pids[@]}"; batch_pids=(); slot=0
        fi
    done
    [[ ${#batch_pids[@]} -gt 0 ]] && wait "${batch_pids[@]}"
    echo "=== TP=2 complete ==="
fi

# TP=4: sequential — all 4 cards needed per model
if [[ ${#tp4_models[@]} -gt 0 ]]; then
    echo "=== TP=4: ${#tp4_models[@]} model(s) — sequential (all 4 cards) ==="
    for m in "${tp4_models[@]}"; do
        run_model_suite "${m}" 0 "${BASE_PORT}"
    done
    echo "=== TP=4 complete ==="
fi

echo "========================================================"
echo "Sweep complete. Logs in: ${LOG_DIR}/"
echo ""
echo "Summary files:"
ls "${LOG_DIR}"/*.summary.txt 2>/dev/null || echo "  (none found)"
echo "========================================================"
