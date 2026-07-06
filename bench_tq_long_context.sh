#!/bin/bash
# bench_tq_long_context.sh — Argument 1 (Capacity): Long-context on B70.
#
# Demonstrates that TurboQuant enables 16K–32K context serving on Intel B70
# where BF16 either OOMs or is severely concurrency-limited.
#
# For each (model, context_length, config), sweep concurrency to find the
# maximum achievable concurrency. BF16 is included as a canary: if the
# server can start, we measure its ceiling; if it OOMs at init, we record
# that as "OOM" — making the TQ advantage visible.
#
# Per-model TP: Gemma-4-E4B and Llama-3.1-8B use TP=1, Qwen2.5-14B uses TP=2.
# Scheduling: TP=1 models run in PARALLEL, then TP=2 sequentially.
#
# ISL/OSL are scaled with context length so requests actually stress long KV:
#   16K context → ISL=4096, OSL=2048
#   32K context → ISL=8192, OSL=4096
#
# Usage:
#   bash bench_tq_long_context.sh                          # all 3 models
#   bash bench_tq_long_context.sh llama31_8b               # one model
#
# Output:
#   /workspace/bench_results/long_context_<TS>.csv
set -euo pipefail

CONTAINER="vllm-test"
BASE_PORT=8220
CONTEXT_LENGTHS=(16384 32768)

# BF16 included as canary — will OOM or hit low ceiling at long context
CONFIGS=("bf16" "turboquant_4bit_nc")

# Concurrency sweep — lower ceiling than short-context since KV is large
CONCURRENCIES=(1 2 4 8 16 32 64)

RESULT_DIR="/workspace/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV_OUT="${RESULT_DIR}/long_context_${TIMESTAMP}.csv"

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

# ISL/OSL scaled to context length
resolve_isl() { echo $(( $1 / 4 )); }
resolve_osl() { echo $(( $1 / 8 )); }

# ---------------------------------------------------------------------------
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
    local server_log="/tmp/vllm_longctx_server_${model_short}.log"

    echo "[${model_short}] Starting: config=${config} TP=${tp_size} GPU=${gpu} port=${port} ctx=${ctx_len}"

    local kv_arg=""
    [[ "${config}" != "bf16" ]] && kv_arg="--kv-cache-dtype ${config}"

    # max-num-batched-tokens must be >= max_model_len for long context
    local max_batched=${ctx_len}
    [[ ${max_batched} -lt 8192 ]] && max_batched=8192

    local serve_cmd="ZE_AFFINITY_MASK=${gpu} \
        VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
        vllm serve ${model} \
        --port ${port} \
        --tensor-parallel-size ${tp_size} \
        --dtype bfloat16 \
        --max-model-len ${ctx_len} \
        --gpu-memory-utilization 0.92 \
        --enforce-eager \
        --max-num-batched-tokens ${max_batched} \
        --block-size 64 \
        --no-enable-log-requests \
        --no-enable-prefix-caching \
        ${kv_arg} \
        > ${server_log} 2>&1"

    docker exec -d ${CONTAINER} bash -c "${serve_cmd}"

    local attempts=0
    while ! docker exec ${CONTAINER} curl -sf http://localhost:${port}/health > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ ${attempts} -gt 600 ]]; then
            # Check if process already exited (OOM at init)
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
        # Early OOM detection — if process died before health check, bail fast
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

    # Warmup with small request
    local isl=$(resolve_isl "${ctx_len}")
    docker exec ${CONTAINER} vllm bench serve \
        --model "${model}" --backend openai --endpoint /v1/completions \
        --port "${port}" --dataset-name random \
        --random-input-len 64 --random-output-len 32 \
        --num-prompts 2 --max-concurrency 1 \
        --ignore-eos --disable-tqdm > /dev/null 2>&1 || true
}

stop_server() {
    local model_short="$1" port="$2"
    echo "[${model_short}]   Stopping server on port ${port}..."
    docker exec ${CONTAINER} bash -c "
        pkill -TERM -f 'vllm serve.*--port ${port}' 2>/dev/null || true
        sleep 3
        pkill -KILL -f 'vllm serve.*--port ${port}' 2>/dev/null || true
        sleep 2
        if ! pgrep -f 'vllm serve' > /dev/null 2>&1; then
            pkill -KILL -f 'EngineCore' 2>/dev/null || true
            pkill -KILL -f 'from multiprocessing.spawn' 2>/dev/null || true
            rm -f /dev/shm/psm_* /dev/shm/sem.loky-* 2>/dev/null || true
        fi
        for i in \$(seq 1 30); do
            ss -tln 2>/dev/null | grep -q ':${port} ' || break
            sleep 1
        done
    " || true
    sleep 5
}

# ---------------------------------------------------------------------------
run_point() {
    local model_short="$1" config="$2" conc="$3" model="$4" port="$5" ctx_len="$6"
    local isl=$(resolve_isl "${ctx_len}")
    local osl=$(resolve_osl "${ctx_len}")

    # Fewer prompts at long context — each takes much longer
    local n_prompts=$(( conc * 2 ))
    [[ ${n_prompts} -lt 16 ]] && n_prompts=16
    [[ ${n_prompts} -gt 256 ]] && n_prompts=256

    local tag="longctx_${model_short}_${config}_ctx${ctx_len}_c${conc}_${TIMESTAMP}"
    echo "[${model_short}]   ctx=${ctx_len} conc=${conc} isl=${isl} osl=${osl} n=${n_prompts}"

    local bench_log="/tmp/bench_${tag}.log"
    docker exec ${CONTAINER} vllm bench serve \
        --model "${model}" --backend openai --endpoint /v1/completions \
        --port "${port}" --dataset-name random \
        --random-input-len "${isl}" --random-output-len "${osl}" \
        --num-prompts "${n_prompts}" \
        --max-concurrency "${conc}" \
        --ignore-eos --disable-tqdm \
        --save-result --result-dir "${RESULT_DIR}" \
        --result-filename "${tag}.json" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --metadata model="${model}" config="${config}" \
                   context_length="${ctx_len}" concurrency="${conc}" \
                   isl="${isl}" osl="${osl}" \
        > "${bench_log}" 2>&1

    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[${model_short}]     FAILED (rc=${rc}) — likely OOM under load"
        # Check if server is still alive
        if ! docker exec ${CONTAINER} curl -sf http://localhost:${port}/health > /dev/null 2>&1; then
            echo "[${model_short}]     Server died — stopping sweep for this config"
            return 2  # Signal to stop the concurrency sweep
        fi
        return 1
    fi
    return 0
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
                local max_conc_ok=0
                for conc in "${CONCURRENCIES[@]}"; do
                    run_point "${model_short}" "${config}" "${conc}" "${model}" "${port}" "${ctx_len}"
                    local rc=$?
                    if [[ ${rc} -eq 0 ]]; then
                        max_conc_ok=${conc}
                    elif [[ ${rc} -eq 2 ]]; then
                        echo "[${model_short}]   Server crashed — max conc was ${max_conc_ok}"
                        break
                    fi
                    # Don't early-exit on rc=1 (bench failed but server alive) — keep trying
                done
                echo "[${model_short}]   Max successful concurrency: ${max_conc_ok}"
            else
                echo "[${model_short}]   Server failed to start — recording OOM"
                # Record OOM marker
                echo "${model_short},${config},${ctx_len},0,OOM,0,0,0,0,0,0" \
                    >> "/tmp/longctx_raw_${TIMESTAMP}.csv"
            fi
            stop_server "${model_short}" "${port}"
        done
    done
}

# ---------------------------------------------------------------------------
emit_csv() {
    docker exec ${CONTAINER} mkdir -p "${RESULT_DIR}"
    docker exec ${CONTAINER} bash -c "echo 'model,config,context_length,concurrency,num_prompts,request_throughput,output_throughput,mean_ttft_ms,mean_tpot_ms,p99_ttft_ms,p99_tpot_ms,status' > ${CSV_OUT}"

    for ms in "${MODEL_SHORTS[@]}"; do
        for ctx_len in "${CONTEXT_LENGTHS[@]}"; do
            for config in "${CONFIGS[@]}"; do
                # Check if OOM was recorded for this combo
                if grep -q "^${ms},${config},${ctx_len},0,OOM" "/tmp/longctx_raw_${TIMESTAMP}.csv" 2>/dev/null; then
                    docker exec ${CONTAINER} bash -c \
                        "echo '${ms},${config},${ctx_len},0,0,0,0,0,0,0,0,OOM' >> ${CSV_OUT}"
                    continue
                fi
                for conc in "${CONCURRENCIES[@]}"; do
                    local tag="longctx_${ms}_${config}_ctx${ctx_len}_c${conc}_${TIMESTAMP}"
                    docker exec -i ${CONTAINER} python3 - <<PY > /dev/null 2>&1 || true
import json, os
p = "${RESULT_DIR}/${tag}.json"
out = "${CSV_OUT}"
if os.path.exists(p):
    with open(p) as f:
        d = json.load(f)
    completed = d.get('completed', 0)
    status = "OK" if completed > 0 else "FAIL"
    row = ",".join(str(x) for x in [
        "${ms}", "${config}", ${ctx_len}, ${conc},
        d.get("num_prompts", ""),
        f"{d.get('request_throughput',0):.4f}",
        f"{d.get('output_throughput',0):.2f}",
        f"{d.get('mean_ttft_ms',0):.2f}",
        f"{d.get('mean_tpot_ms',0):.2f}",
        f"{d.get('p99_ttft_ms',0):.2f}",
        f"{d.get('p99_tpot_ms',0):.2f}",
        status,
    ])
    with open(out, "a") as f:
        f.write(row + "\n")
PY
                done
            done
        done
    done

    echo ""
    echo "CSV: ${CSV_OUT}"
    docker exec ${CONTAINER} cat "${CSV_OUT}"
}

# ===========================================================================
echo "============================================"
echo "  Long-Context Capacity Benchmark"
echo "  Models:     ${MODEL_SHORTS[*]}"
echo "  Configs:    ${CONFIGS[*]}"
echo "  Context:    ${CONTEXT_LENGTHS[*]}"
echo "  Conc:       ${CONCURRENCIES[*]}"
echo "  Timestamp:  ${TIMESTAMP}"
echo "  BF16 included as canary (expect OOM at long context)"
echo "  TP=1 models run in parallel on separate GPUs"
echo "============================================"

docker exec ${CONTAINER} mkdir -p ${RESULT_DIR}
: > "/tmp/longctx_raw_${TIMESTAMP}.csv"

# Split models by TP
TP1_MODELS=()
TP2_MODELS=()
for ms in "${MODEL_SHORTS[@]}"; do
    tp=$(resolve_tp "${ms}")
    if [[ "${tp}" == "1" ]]; then
        TP1_MODELS+=("${ms}")
    else
        TP2_MODELS+=("${ms}")
    fi
done

FAIL=0

# --- Phase 1: TP=1 models in parallel on separate GPUs ---
if [[ ${#TP1_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo ">>> Phase 1: TP=1 models in parallel (${TP1_MODELS[*]})"
    PIDS=()
    for ms in "${TP1_MODELS[@]}"; do
        run_model_suite "${ms}" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}" || FAIL=1
    done
    echo ""
    echo ">>> Phase 1 complete"
fi

# --- Phase 2: TP=2 models sequentially (need both GPUs) ---
if [[ ${#TP2_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo ">>> Phase 2: TP=2 models sequentially (${TP2_MODELS[*]})"
    docker exec ${CONTAINER} bash -c "
        rm -f /dev/shm/psm_* /dev/shm/sem.loky-* 2>/dev/null || true
    " || true
    sleep 10
    for ms in "${TP2_MODELS[@]}"; do
        run_model_suite "${ms}" || FAIL=1
    done
    echo ""
    echo ">>> Phase 2 complete"
fi

emit_csv
exit ${FAIL}
