#!/bin/bash
# bench_aiperf.sh — SLA-bound max concurrency via NVIDIA AIPerf (github.com/ai-dynamo/aiperf),
# replaying multi-turn agentic coding trajectories (AA-AgentPerf-style).
#
# Same model registry and server lifecycle as bench_fp8_kv_sla_concurrency.sh, but load is
# a multi-turn agentic-code trace (AIPerf's `aiperf synthesize agentic-code` generator) rather
# than uniform single-turn requests. The trace models shared global/session prefix layers,
# per-session context growth turn over turn, and inter-turn delays — the same shape AA-AgentPerf
# measures — and is replayed via `--custom-dataset-type mooncake_trace`. The concurrency search
# uses AIPerf's `max-concurrency-under-sla` recipe to find the max concurrent agent sessions the
# server sustains before TTFT/ITL breaks SLA — this IS the memory/GPU bottleneck signal we care
# about, so each model is served exactly ONCE with vLLM's default (bf16) KV cache — no bf16-vs-fp8
# KV comparison here; that's what the other bench_fp8_kv_*.sh scripts already do. Prefix caching
# is left ENABLED (unlike the other fp8-kv scripts) since the trace's KV-cache-reuse layers are
# only meaningful with caching on. Before any model is served, `aiperf analyze-trace` validates
# the generated dataset (cache hit rate, ISL/OSL shape, prefix reuse) so we confirm the trace is
# actually agentic-shaped before spending GPU time on it. The dataset's --max-isl is capped to
# MAX_MODEL_LEN so no turn ever exceeds vLLM's context window and gets rejected mid-sweep. AIPerf
# writes its own profile_export_aiperf.{csv,json} per run — this script only handles server
# lifecycle, dataset generation/validation, and artifact collection. GPU memory/utilization
# telemetry during the run is NOT captured (AIPerf's --gpu-telemetry only supports NVIDIA/AMD,
# not Intel XPU) — a separate xpu-smi-based sampler would be needed for that and is intentionally
# left out for now.
#
# SLA: TTFT(p90) < SLA_TTFT_MS  AND  ITL(p90) < SLA_ITL_MS (ITL is AIPerf's per-token-latency
# analogue of TPOT — lower is better, so it uses the same "lt" comparator as TTFT).
#
# Usage:
#   bash bench_aiperf.sh                          # all 11 models
#   bash bench_aiperf.sh llama31 qwen25           # specific models
#   bash bench_aiperf.sh --num-sessions 200       # smaller/larger agentic trace
#   bash bench_aiperf.sh --sla-ttft-ms 3000 --sla-itl-ms 100 qwen3
#
# Output:
#   /tmp/aiperf_bench/dataset/                          — generated agentic trace (shared across models)
#   /tmp/aiperf_bench/<model>_tp<TP>_<TS>/profile_export_aiperf.{csv,json}
#   Logs: /tmp/aiperf_bench/logs/
set -euo pipefail

# ── Environment (required for Intel XPU backend) ────────────────────────────
export VLLM_TARGET_DEVICE=xpu
export VLLM_MLA_DISABLE=1
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT=900
export VLLM_NO_USAGE_STATS=1
export HF_HOME=/tmp   # container /tmp is bind-mounted to ~/LLM on host

BASE_PORT=8280
# Kept in sync with ensure_agentic_dataset()'s --max-isl so vLLM never rejects a turn for
# exceeding context length (the agentic-code generator otherwise targets ~167K-token sessions).
MAX_MODEL_LEN=32000
# --max-isl caps input tokens only; a turn's total context is input+output, so cap the dataset
# below MAX_MODEL_LEN to leave headroom for the generator's output tokens (mean ~1000, max ~1500).
AGENTIC_MAX_ISL=$(( MAX_MODEL_LEN - 2000 ))

SLA_TTFT_MS=5000
SLA_ITL_MS=200          # AIPerf inter-token latency — analogue of TPOT

CONCURRENCY_MIN=1
CONCURRENCY_MAX=256
WARMUP_S=60
DURATION_S=300

NUM_SESSIONS=500        # agentic-code trace: number of simulated coding-agent sessions
DATASET_SEED=42

RESULT_DIR="/tmp/aiperf_bench"
LOG_DIR="${RESULT_DIR}/logs"
DATASET_DIR="${RESULT_DIR}/dataset"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATASET_FILE=""          # populated by ensure_agentic_dataset()

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

# Returns space-separated list of TP values for this model.
resolve_tp_list() {
    case "$1" in
        llama33_70b|qwen25_72b|deepseekr1_70b) echo "4" ;;
        gemma4|qwen25) echo "1 2" ;;
        mistral)       echo "2" ;;
        gemma4_31b)    echo "2" ;;
        *)             echo "1" ;;
    esac
}

# Extra vllm serve flags per model.
resolve_extra_args() {
    case "$1" in
        deepseekr1|deepseekr1_70b) echo "--trust-remote-code --quantization fp8" ;;
        gemma4)         echo "--trust-remote-code --attention-backend TRITON_ATTN --quantization fp8" ;;
        gemma4_31b)     echo "--trust-remote-code --attention-backend FLASH_ATTN --quantization fp8" ;;
        *)              echo "--quantization fp8" ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-sessions)  NUM_SESSIONS="$2"; shift 2 ;;
        --seed)          DATASET_SEED="$2"; shift 2 ;;
        --sla-ttft-ms)   SLA_TTFT_MS="$2"; shift 2 ;;
        --sla-itl-ms)    SLA_ITL_MS="$2"; shift 2 ;;
        --concurrency-min) CONCURRENCY_MIN="$2"; shift 2 ;;
        --concurrency-max) CONCURRENCY_MAX="$2"; shift 2 ;;
        --duration-s)    DURATION_S="$2"; shift 2 ;;
        *)               break ;;
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
# AIPerf must be installed to drive the benchmark; install on first use.
ensure_aiperf() {
    if ! command -v aiperf > /dev/null 2>&1; then
        echo "aiperf not found — installing (pip install aiperf)..."
        pip install --quiet aiperf
    fi
    aiperf --version
}

# Generates the multi-turn agentic-code trace once, shared across every model/config run below
# (the trace is a token-count-level Mooncake JSONL — model/tokenizer only matter at replay time).
ensure_agentic_dataset() {
    local signature="sessions=${NUM_SESSIONS};seed=${DATASET_SEED};max_isl=${AGENTIC_MAX_ISL};config=multiturn_v1"
    local signature_file="${DATASET_DIR}/.gen_signature"
    local existing
    existing=$(find "${DATASET_DIR}" -maxdepth 1 -name "dataset.jsonl" 2>/dev/null | head -1 || true)
    if [[ -n "${existing}" && -f "${signature_file}" && "$(cat "${signature_file}")" == "${signature}" ]]; then
        echo "Reusing existing agentic-code trace (params unchanged): ${existing}"
        DATASET_FILE="${existing}"
        return
    fi
    [[ -n "${existing}" ]] && echo "Existing trace found but generation params changed — regenerating..."

    echo "Generating agentic-code trace (${NUM_SESSIONS} sessions, seed=${DATASET_SEED}, max-isl=${AGENTIC_MAX_ISL})..."
    mkdir -p "${DATASET_DIR}"

    # Custom config: scale cache layers to fit within MAX_MODEL_LEN and ensure multi-turn growth.
    # Default config has layer1=32K which alone exceeds our max-isl, collapsing to single-turn.
    local config_file="${DATASET_DIR}/agentic_config.json"
    cat > "${config_file}" <<CFGEOF
{
  "new_tokens_per_turn": {"mean": 3000, "median": 2500, "max": 6000, "bias": 1.1},
  "generation_length": {"mean": 800, "median": 600, "max": 1500},
  "inter_turn_delay": {
    "agentic_fraction": 0.7,
    "agentic_delay": {"mean": 2500, "median": 1800, "max": 10000},
    "human_delay": {"mean": 40000, "median": 25000, "max": 120000},
    "max": null
  },
  "turns": {"mean": 5, "median": 4, "min": 2, "max": 12},
  "max_prompt_tokens": ${AGENTIC_MAX_ISL},
  "block_size": 64,
  "cache": {
    "layer1_tokens": 4000,
    "layer1_5_tokens": 2000,
    "layer2": {"mean": 3000, "median": 2000, "max": 6000},
    "layer1_5_groups": {"num_groups": 50, "zipf_alpha": 1.2}
  }
}
CFGEOF

    aiperf synthesize agentic-code \
        --num-sessions "${NUM_SESSIONS}" \
        --seed "${DATASET_SEED}" \
        --max-isl "${AGENTIC_MAX_ISL}" \
        --config "${config_file}" \
        --output "${DATASET_DIR}" \
        > "${LOG_DIR}/synthesize_agentic_code.log" 2>&1

    # `synthesize` writes into a timestamped subdirectory — flatten to a stable path.
    local generated
    generated=$(find "${DATASET_DIR}" -mindepth 2 -maxdepth 2 -name "dataset.jsonl" | sort | tail -1)
    if [[ -z "${generated}" ]]; then
        echo "ERROR: agentic-code dataset generation failed — see ${LOG_DIR}/synthesize_agentic_code.log"
        exit 1
    fi
    cp "${generated}" "${DATASET_DIR}/dataset.jsonl"
    echo "${signature}" > "${signature_file}"
    DATASET_FILE="${DATASET_DIR}/dataset.jsonl"
    echo "Dataset ready: ${DATASET_FILE}"
}

# Validates the trace actually has agentic shape (context growth, prefix reuse) before any
# GPU time is spent on it — prints AIPerf's cache-hit-rate / ISL-OSL report and saves it to JSON.
validate_agentic_dataset() {
    local analysis_file="${DATASET_DIR}/trace_analysis.json"
    echo ""
    echo "Validating agentic-code trace (cache hit rate, ISL/OSL shape)..."
    aiperf analyze-trace "${DATASET_FILE}" \
        --block-size 64 \
        --output-file "${analysis_file}" \
        | tee "${LOG_DIR}/analyze_trace.log"
    echo "Trace analysis saved: ${analysis_file}"
}

# ---------------------------------------------------------------------------
start_server() {
    local model_short="$1" model="$2" extra_args="$3" card="$4" port="$5" tp="$6"
    local server_log="${LOG_DIR}/vllm_server_${model_short}_tp${tp}.log"

    echo "[${model_short}] Starting: tp=${tp} card=${card} port=${port}"

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

    local serve_cmd="VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
        vllm serve ${model} \
        --port ${port} \
        --tensor-parallel-size 4 \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization 0.92 \
        --enforce-eager \
        --max-num-batched-tokens 8192 \
        --max-num-seq 256 \
        --block-size 64 \
        --no-enable-log-requests \
        ${extra_args} \
        > ${server_log} 2>&1"

    setsid bash -c "${serve_cmd}" &
    local server_pid=$!
    echo "${server_pid}" > "${LOG_DIR}/vllm_pid_${model_short}.txt"

    echo "[${model_short}]   Waiting for server (up to 900s)..."
    local attempts=0
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
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
}

stop_server() {
    local model_short="$1" port="$2"
    echo "[${model_short}]   Stopping server on port ${port}..."
    local pid_file="${LOG_DIR}/vllm_pid_${model_short}.txt"
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
    for i in $(seq 1 30); do
        ss -tln 2>/dev/null | grep -q ":${port} " || break
        sleep 1
    done
    sleep 3
}

# ---------------------------------------------------------------------------
run_aiperf() {
    local model_short="$1" model="$2" port="$3" tp="$4"
    local artifact_dir="${RESULT_DIR}/${model_short}_tp${tp}_${TIMESTAMP}"
    local run_log="${LOG_DIR}/aiperf_${model_short}_tp${tp}.log"

    echo "[${model_short}]   Running AIPerf (agentic trace, sessions=${NUM_SESSIONS} conc=[${CONCURRENCY_MIN},${CONCURRENCY_MAX}])..."
    echo "[${model_short}]     Warmup: ${WARMUP_S}s | Profiling duration: ${DURATION_S}s per concurrency level"
    echo "[${model_short}]     SLA targets: TTFT < ${SLA_TTFT_MS}ms, ITL < ${SLA_ITL_MS}ms"
    echo "[${model_short}]     Log (tail -f): ${run_log}"

    local rc=0
    aiperf profile \
        --model "${model}" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --tokenizer "${model}" \
        --url "localhost:${port}" \
        --input-file "${DATASET_FILE}" \
        --custom-dataset-type mooncake_trace \
        --no-fixed-schedule \
        --search-recipe max-concurrency-under-sla \
        --ttft-sla-ms "${SLA_TTFT_MS}" \
        --itl-sla-ms "${SLA_ITL_MS}" \
        --concurrency-min "${CONCURRENCY_MIN}" \
        --concurrency-max "${CONCURRENCY_MAX}" \
        --warmup-duration "${WARMUP_S}" \
        --benchmark-duration "${DURATION_S}" \
        --output-artifact-dir "${artifact_dir}" \
        2>&1 | tee "${run_log}" | grep --line-buffered -E "Phase|NOTICE|search_iter|concurrency|SLA|result|complete|ERROR" || rc=${PIPESTATUS[0]}

    if [[ ${rc} -ne 0 ]]; then
        echo "[${model_short}]     AIPerf FAILED (rc=${rc}) — see ${run_log}"
        tail -30 "${run_log}" 2>/dev/null || true
        return 1
    fi

    echo "[${model_short}]     AIPerf complete — artifacts in ${artifact_dir}"
}

# ---------------------------------------------------------------------------
run_model_suite() {
    local model_short="$1" card="${2:-0}" port="${3:-${BASE_PORT}}"
    local model extra_args tp_list
    model=$(resolve_model "${model_short}")
    extra_args=$(resolve_extra_args "${model_short}")
    tp_list=$(resolve_tp_list "${model_short}")

    echo ""
    echo "[${model_short}] ========================================"
    echo "[${model_short}]  Model: ${model}  Card: ${card}  Port: ${port}"
    echo "[${model_short}]  TP configs: ${tp_list}"
    echo "[${model_short}] ========================================"

    for tp in ${tp_list}; do
        echo ""
        echo "[${model_short}] --- tp=${tp} ---"
        if start_server "${model_short}" "${model}" "${extra_args}" "${card}" "${port}" "${tp}"; then
            run_aiperf "${model_short}" "${model}" "${port}" "${tp}" || true
        else
            echo "[${model_short}]   Server failed to start — skipping AIPerf run"
        fi
        stop_server "${model_short}" "${port}"
    done
}

# ===========================================================================
echo "============================================"
echo "  AIPerf SLA-bound Max Concurrency Benchmark"
echo "  Models:   ${MODEL_SHORTS[*]}"
echo "  SLA:      TTFT(p99)<${SLA_TTFT_MS}ms  AND  ITL(p99)<${SLA_ITL_MS}ms"
echo "  Dataset:  agentic-code trace, ${NUM_SESSIONS} sessions, seed=${DATASET_SEED}, max_model_len=${MAX_MODEL_LEN}"
echo "  Timestamp: ${TIMESTAMP}"
echo "============================================"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
ensure_aiperf
ensure_agentic_dataset
validate_agentic_dataset

echo "Checking for leftover vllm processes..."
if pgrep -f "vllm serve" > /dev/null 2>&1 || pgrep -f "EngineCore" > /dev/null 2>&1; then
    echo "  Found stale vllm processes — killing before starting benchmark..."
    pkill -TERM -f "vllm serve" 2>/dev/null || true
    sleep 3
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    pkill -KILL -f "EngineCore" 2>/dev/null || true
    sleep 3
else
    echo "  None found. Proceeding."
fi

FAIL=0
for ms in "${MODEL_SHORTS[@]}"; do
    run_model_suite "${ms}" "0" "${BASE_PORT}" || FAIL=1
done

echo ""
echo "================================================================"
echo "  AIPerf artifacts (one dir per model/TP run):"
find "${RESULT_DIR}" -maxdepth 1 -type d -name "*_${TIMESTAMP}" 2>/dev/null
echo "  Each artifact dir contains profile_export_aiperf.csv / .json"
echo "  Logs: ${LOG_DIR}/"
echo "================================================================"

exit ${FAIL}
