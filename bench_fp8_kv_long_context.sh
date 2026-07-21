#!/bin/bash
# bench_fp8_kv_long_context.sh — Long-context capacity benchmark for FP8 KV cache on Intel B70.
#
# Demonstrates that FP8 KV cache enables higher concurrency at 16K–32K context
# on Intel Arc Pro B70 hardware, where BF16 either OOMs or hits a lower ceiling.
#
# BF16 is included as a canary: if the server OOMs at init, it is recorded as "OOM"
# making the FP8 KV cache advantage visible.
#
# Configs:  bf16 (baseline)  vs  fp8 (--kv-cache-dtype fp8)
#
# Per-model TP:
#   TP=1 only:   llama31, deepseekr1, gemma3, qwen3
#   TP=1 + TP=2: gemma4, qwen25
#   TP=2 only:   mistral, gemma4_31b
#   TP=4 only:   llama33_70b, qwen25_72b, deepseekr1_70b
#
# Scheduling:
#   Phase 1: TP=1-only models run in parallel (one per card, up to 4).
#   Phase 2: TP=2-capable models run in parallel pairs (cards 0,1 and 2,3), batched.
#   Phase 3: TP=4 models run sequentially (all 4 cards).
#
# Context lengths:  16384, 32768
# ISL/OSL scaled:   16K → ISL=4096 OSL=2048 | 32K → ISL=8192 OSL=4096
#
# Usage:
#   bash bench_fp8_kv_long_context.sh                         # all 10 models, all phases
#   bash bench_fp8_kv_long_context.sh llama31 qwen25          # specific models
#   bash bench_fp8_kv_long_context.sh --configs fp8 llama31   # fp8 only
#   bash bench_fp8_kv_long_context.sh --phase 1               # TP=1 models only
#   bash bench_fp8_kv_long_context.sh --phase 2               # TP=2 models only
#   bash bench_fp8_kv_long_context.sh --phase 3               # TP=4 models only
#
# Output:
#   /tmp/fp8kv_long_context/long_context_<TS>.csv   (rows appended live; host: ~/LLM/fp8kv_long_context/)
set -euo pipefail

# ── Environment (required for Intel XPU backend) ────────────────────────────
export VLLM_TARGET_DEVICE=xpu
export VLLM_MLA_DISABLE=1
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT=900
export VLLM_NO_USAGE_STATS=1
export HF_HOME=/tmp   # container /tmp is bind-mounted to ~/LLM on host

BASE_PORT=8240
CONTEXT_LENGTHS=(16384 32768)
CONFIGS=("bf16" "fp8")
CONCURRENCIES=(1 2 4 8 16 32 64)

RESULT_DIR="/tmp/fp8kv_long_context"
LOG_DIR="${RESULT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV_OUT="${RESULT_DIR}/long_context_${TIMESTAMP}.csv"
# Host-visible copy: /tmp maps to /home/intel/LLM on the host
CSV_HOST="/tmp/long_context_${TIMESTAMP}.csv"

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
# Parse flags: --configs <c1,c2>  --phase <1|2|3>  --card N  --port N
PHASE=""        # empty = run all phases
FORCE_CARD=""   # when set, bypass phase scheduling and run on this base card
FORCE_PORT=""   # when set, use this port (paired with --card)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs)
            IFS=',' read -ra CONFIGS <<< "$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --card)
            FORCE_CARD="$2"
            shift 2
            ;;
        --port)
            FORCE_PORT="$2"
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
    local model_short="$1" config="$2" model="$3" extra_args="$4" card="$5" port="$6" tp="$7" ctx_len="$8"
    local server_log="${LOG_DIR}/vllm_longctx_server_${model_short}_${config}_tp${tp}_ctx${ctx_len}.log"

    echo "[${model_short}] Starting: config=${config} tp=${tp} card=${card} port=${port} ctx=${ctx_len}"

    local kv_arg=""
    [[ "${config}" != "bf16" ]] && kv_arg="--kv-cache-dtype ${config}"

    # --max-num-batched-tokens deliberately omitted from vllm serve below.
    # Scaling it with ctx_len forces vLLM to profile activation memory for a
    # giant-batch forward pass, starving KV cache at long context (e.g. OOMs
    # at 32K on gemma4_31b TP=2 even though vLLM's own default lets the same
    # model fit 64K on the same B70x2 hardware via chunked prefill). Let vLLM
    # pick its own default instead.

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
        --max-model-len ${ctx_len} \
        --gpu-memory-utilization 0.92 \
        --enforce-eager \
        --max-num-seq 32 \
        --block-size 64 \
        --no-enable-log-requests \
        --no-enable-prefix-caching \
        ${kv_arg} \
        ${extra_args} \
        > ${server_log} 2>&1"

    setsid bash -c "${serve_cmd}" &
    local server_pid=$!
    echo "${server_pid}" > "${LOG_DIR}/vllm_longctx_pid_${model_short}.txt"

    echo "[${model_short}]   Waiting for server (up to 900s)..."
    local attempts=0
    while ! curl -sf http://localhost:${port}/health > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ ${attempts} -gt 900 ]]; then
            echo "[${model_short}]   ERROR: server failed to start after 900s"
            tail -40 "${server_log}" 2>/dev/null || true
            stop_server "${model_short}" "${port}"
            return 1
        fi
        # Early OOM detection every 30s — if the process died, bail fast
        if [[ $((attempts % 30)) -eq 0 ]]; then
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                echo "[${model_short}]   OOM or crash at init (config=${config} tp=${tp} ctx=${ctx_len})"
                tail -20 "${server_log}" 2>/dev/null | grep -iE "error|oom|memory|killed" || true
                return 1
            fi
        fi
        sleep 1
    done
    echo "[${model_short}]   Server ready in ${attempts}s"

    # Warmup with a small request
    vllm bench serve \
        --model "${model}" --backend openai --endpoint /v1/completions \
        --port "${port}" --dataset-name random \
        --random-input-len 64 --random-output-len 32 \
        --num-prompts 2 --max-concurrency 1 \
        --ignore-eos --disable-tqdm > /dev/null 2>&1 || true
}

stop_server() {
    local model_short="$1" port="$2"
    echo "[${model_short}]   Stopping server on port ${port}..."
    local pid_file="${LOG_DIR}/vllm_longctx_pid_${model_short}.txt"
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
    for i in $(seq 1 30); do
        ss -tln 2>/dev/null | grep -q ":${port} " || break
        sleep 1
    done
    sleep 3
}

# ---------------------------------------------------------------------------
run_point() {
    local model_short="$1" config="$2" tp="$3" conc="$4" model="$5" port="$6" ctx_len="$7"
    local isl=$(( ctx_len / 4 ))
    # Cap OSL at 512 — long output is not the goal here; we're measuring KV
    # cache capacity (how many concurrent long-prefill requests fit in VRAM).
    # Uncapped OSL (ctx/8 = 2048–4096) makes each bench point take 20–40 min.
    local osl=512

    # Fewer prompts at long context — prefill alone is expensive
    local n_prompts=$(( conc * 2 ))
    [[ ${n_prompts} -lt 8 ]] && n_prompts=8
    [[ ${n_prompts} -gt 64 ]] && n_prompts=64

    local tag="longctx_${model_short}_${config}_tp${tp}_ctx${ctx_len}_c${conc}_${TIMESTAMP}"
    local bench_log="${LOG_DIR}/bench_${tag}.log"

    echo "[${model_short}]   ctx=${ctx_len} conc=${conc} isl=${isl} osl=${osl} n=${n_prompts}"

    # Per-point timeout: n_prompts × ~60s per prompt max (long context is slow)
    # Prevents any single bench call from hanging indefinitely (e.g. gemma4 bf16 ctx=32K)
    local bench_timeout=$(( n_prompts * 60 ))
    [[ ${bench_timeout} -lt 300 ]] && bench_timeout=300

    local rc=0
    timeout "${bench_timeout}" vllm bench serve \
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
        --metadata model="${model}" config="${config}" tp="${tp}" \
                   context_length="${ctx_len}" concurrency="${conc}" \
                   isl="${isl}" osl="${osl}" \
        > "${bench_log}" 2>&1 || rc=$?

    # Append CSV row immediately after each bench point (both container and host paths)
    python3 - <<PY | tee -a "${CSV_OUT}" >> "${CSV_HOST}" 2>/dev/null || true
import json, os, sys
p = "${RESULT_DIR}/${tag}.json"
if not os.path.exists(p):
    sys.exit(0)
with open(p) as f:
    d = json.load(f)
completed = d.get('completed', 0)
status = "OK" if completed > 0 else "FAIL"
row = ",".join(str(x) for x in [
    "${model_short}", "${config}", ${tp}, ${ctx_len}, ${conc},
    d.get("num_prompts", ""),
    f"{d.get('request_throughput', 0):.4f}",
    f"{d.get('output_throughput', 0):.2f}",
    f"{d.get('mean_ttft_ms', 0):.2f}",
    f"{d.get('mean_tpot_ms', 0):.2f}",
    f"{d.get('p99_ttft_ms', 0):.2f}",
    f"{d.get('p99_tpot_ms', 0):.2f}",
    status,
])
print(row)
PY

    if [[ ${rc} -ne 0 ]]; then
        echo "[${model_short}]     FAILED (rc=${rc})"
        if ! curl -sf http://localhost:${port}/health > /dev/null 2>&1; then
            echo "[${model_short}]     Server died — stopping sweep for this config"
            return 2  # Signal caller to stop concurrency sweep
        fi
        return 1
    fi
    return 0
}

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
        for ctx_len in "${CONTEXT_LENGTHS[@]}"; do
            for config in "${CONFIGS[@]}"; do
                echo ""
                echo "[${model_short}] --- tp=${tp} ctx=${ctx_len} config=${config} ---"

                # Attempt to start the server; on OOM retry with --quantization fp8.
                local effective_args="${extra_args}"
                local server_started=0
                if start_server "${model_short}" "${config}" "${model}" "${effective_args}" \
                                 "${card}" "${port}" "${tp}" "${ctx_len}"; then
                    server_started=1
                elif [[ "${effective_args}" != *"--quantization fp8"* ]]; then
                    echo "[${model_short}]   Retrying with --quantization fp8 (weights) to reduce VRAM..."
                    stop_server "${model_short}" "${port}"
                    effective_args="${effective_args} --quantization fp8"
                    if start_server "${model_short}" "${config}" "${model}" "${effective_args}" \
                                     "${card}" "${port}" "${tp}" "${ctx_len}"; then
                        server_started=1
                        echo "[${model_short}]   Retry succeeded with --quantization fp8"
                    fi
                fi

                if [[ ${server_started} -eq 1 ]]; then
                    local max_conc_ok=0
                    for conc in "${CONCURRENCIES[@]}"; do
                        local rc=0
                        run_point "${model_short}" "${config}" "${tp}" "${conc}" \
                                  "${model}" "${port}" "${ctx_len}" || rc=$?
                        if [[ ${rc} -eq 0 ]]; then
                            max_conc_ok=${conc}
                        elif [[ ${rc} -eq 2 ]]; then
                            echo "[${model_short}]   Server crashed — max conc was ${max_conc_ok}"
                            break
                        fi
                        # rc=1: bench failed but server alive — stop sweep (likely OOM under load)
                        [[ ${rc} -ne 0 ]] && break
                    done
                    echo "[${model_short}]   Max successful concurrency: ${max_conc_ok}"
                else
                    echo "[${model_short}]   Server failed to start (OOM even with --quantization fp8) — recording OOM"
                    local oom_row="${model_short},${config},${tp},${ctx_len},0,0,0,0,0,0,0,0,OOM"
                    echo "${oom_row}" >> "${CSV_OUT}"
                    echo "${oom_row}" >> "${CSV_HOST}"
                fi
                stop_server "${model_short}" "${port}"
            done
        done
    done
}

# ---------------------------------------------------------------------------
emit_csv() {
    # Rows are already appended live during the run; just print the final CSV.
    echo ""
    echo "CSV (container): ${CSV_OUT}"
    echo "CSV (host):      ${CSV_HOST}"
    cat "${CSV_OUT}"
}

# ===========================================================================
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# Kill any leftover vllm processes from previous runs
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

# Split models by max TP to determine scheduling phase
TP1_MODELS=()
TP2_MODELS=()
TP4_MODELS=()
for ms in "${MODEL_SHORTS[@]}"; do
    tp_list=$(resolve_tp_list "${ms}")
    max_tp=$(echo "${tp_list}" | tr ' ' '\n' | sort -n | tail -1)
    if [[ "${max_tp}" == "1" ]]; then
        TP1_MODELS+=("${ms}")
    elif [[ "${max_tp}" == "2" ]]; then
        TP2_MODELS+=("${ms}")
    else
        TP4_MODELS+=("${ms}")
    fi
done

FAIL=0

# Print banner now that we know which models belong to the selected phase
case "${PHASE}" in
    1) PHASE_MODELS=("${TP1_MODELS[@]+"${TP1_MODELS[@]}"}") ;;
    2) PHASE_MODELS=("${TP2_MODELS[@]+"${TP2_MODELS[@]}"}") ;;
    3) PHASE_MODELS=("${TP4_MODELS[@]+"${TP4_MODELS[@]}"}") ;;
    *) PHASE_MODELS=("${MODEL_SHORTS[@]}") ;;
esac

echo "============================================"
echo "  FP8 KV Cache Long-Context Capacity Benchmark"
[[ -n "${PHASE}" ]] && echo "  Phase:     ${PHASE}"
echo "  Models:    ${PHASE_MODELS[*]:-none}"
echo "  Configs:   ${CONFIGS[*]}"
echo "  Context:   ${CONTEXT_LENGTHS[*]}"
echo "  Conc:      ${CONCURRENCIES[*]}"
echo "  Timestamp: ${TIMESTAMP}"
echo "  BF16 included as canary (expect OOM at long context)"
echo "============================================"

# Write CSV header once at startup — rows are appended live as each point completes.
HEADER="model,config,tp,context_length,concurrency,num_prompts,request_throughput,output_throughput,mean_ttft_ms,mean_tpot_ms,p99_ttft_ms,p99_tpot_ms,status"
echo "${HEADER}" > "${CSV_OUT}"
echo "${HEADER}" > "${CSV_HOST}"

run_phase1() {
    [[ ${#TP1_MODELS[@]} -eq 0 ]] && return 0
    echo ""
    echo ">>> Phase 1: TP=1 models in parallel (${TP1_MODELS[*]})"
    local pids=()
    local i=0
    for ms in "${TP1_MODELS[@]}"; do
        local card="${i}"
        local port=$((BASE_PORT + i))
        run_model_suite "${ms}" "${card}" "${port}" &
        pids+=($!)
        i=$((i + 1))
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || FAIL=1
    done
    echo ">>> Phase 1 complete"
}

run_phase2() {
    [[ ${#TP2_MODELS[@]} -eq 0 ]] && return 0
    echo ""
    echo ">>> Phase 2: TP=2 models in parallel pairs (${TP2_MODELS[*]})"
    sleep 5
    # Pre-assign models to 2 slots via round-robin so each slot runs its
    # models sequentially while the two slots run in parallel.
    # This avoids stalling slot 1 while waiting for slot 0 to finish a batch.
    local slot0_models=()
    local slot1_models=()
    local i=0
    for ms in "${TP2_MODELS[@]}"; do
        if (( i % 2 == 0 )); then
            slot0_models+=("${ms}")
        else
            slot1_models+=("${ms}")
        fi
        i=$(( i + 1 ))
    done

    run_slot() {
        local slot="${1}"; shift
        local models=("$@")
        local card=$(( slot * 2 ))
        local port=$(( BASE_PORT + slot ))
        for ms in "${models[@]}"; do
            run_model_suite "${ms}" "${card}" "${port}" || return 1
        done
    }

    local slot_pids=()
    run_slot 0 "${slot0_models[@]}" &
    slot_pids+=($!)
    if [[ ${#slot1_models[@]} -gt 0 ]]; then
        run_slot 1 "${slot1_models[@]}" &
        slot_pids+=($!)
    fi
    for pid in "${slot_pids[@]}"; do
        wait "${pid}" || FAIL=1
    done
    echo ">>> Phase 2 complete"
}

run_phase3() {
    [[ ${#TP4_MODELS[@]} -eq 0 ]] && return 0
    echo ""
    echo ">>> Phase 3: TP=4 models sequentially (${TP4_MODELS[*]})"
    sleep 5
    for ms in "${TP4_MODELS[@]}"; do
        run_model_suite "${ms}" "0" "${BASE_PORT}" || FAIL=1
    done
    echo ">>> Phase 3 complete"
}

# --card/--port: bypass phase scheduling, run listed models sequentially on fixed card/port
if [[ -n "${FORCE_CARD}" ]]; then
    _port="${FORCE_PORT:-${BASE_PORT}}"
    echo ">>> Direct run: models=(${MODEL_SHORTS[*]}) card=${FORCE_CARD} port=${_port}"
    for ms in "${MODEL_SHORTS[@]}"; do
        run_model_suite "${ms}" "${FORCE_CARD}" "${_port}" || FAIL=1
    done
else
    case "${PHASE}" in
        1) run_phase1 ;;
        2) run_phase2 ;;
        3) run_phase3 ;;
        "") run_phase1; run_phase2; run_phase3 ;;
        *) echo "Unknown --phase value: ${PHASE}. Use 1, 2, or 3."; exit 1 ;;
    esac
fi

emit_csv
exit ${FAIL}
