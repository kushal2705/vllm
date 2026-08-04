#!/bin/bash
# Paired thinking-mode benchmark for google/gemma-4-31B-it on Intel XPU.
#
# Compares Gemma 4 thinking ON and OFF on the same MATH-500 prompts and seeds.
# Reports answer accuracy, latency, total/reasoning/answer token counts, and
# throughput. The server's usage count is exact for total generated tokens;
# reasoning and answer text are re-tokenized for attribution, with the residual
# reported as control tokens (channel delimiters and other special tokens).
#
# Usage:
#   bash bench_gemma4_thinking.sh
#   LIMIT=100 TP=2 MAX_COMPLETION_TOKENS=8192 bash bench_gemma4_thinking.sh
#   THINKING_BUDGET=4096 bash bench_gemma4_thinking.sh
#
# Runs directly inside the vLLM container. The model must already be accessible
# from Hugging Face. Override PYTHON only when the container uses another path.
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-31B-it}"
PORT="${PORT:-8260}"
TP="${TP:-2}"
LIMIT="${LIMIT:-50}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
THINKING_BUDGET="${THINKING_BUDGET:-}"
SEED="${SEED:-42}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
PYTHON="${PYTHON:-python3}"
RESULT_DIR="${RESULT_DIR:-/tmp/gemma4_thinking}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULT_DIR}/${TIMESTAMP}"
SERVER_LOG="${RUN_DIR}/server.log"
REQUESTS_JSONL="${RUN_DIR}/requests.jsonl"
SUMMARY_JSON="${RUN_DIR}/summary.json"
SUMMARY_CSV="${RUN_DIR}/summary.csv"

mkdir -p "${RUN_DIR}"

if [[ "${CONCURRENCY}" != "1" ]]; then
    echo "CONCURRENCY=${CONCURRENCY} is not supported by this paired sequential benchmark." >&2
    echo "Use CONCURRENCY=1; run a separate load benchmark for concurrent throughput." >&2
    exit 1
fi

if [[ "${PYTHON}" != */* ]]; then
    PYTHON="$(command -v "${PYTHON}" || true)"
fi

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "Python interpreter not found. This script expects python3 in the container." >&2
    echo "Override it with PYTHON=/path/to/python if needed." >&2
    exit 1
fi

if ! "${PYTHON}" -c "import aiohttp, datasets, math_verify, transformers"; then
    echo "Missing benchmark dependencies. Install them with:" >&2
    echo "  ${PYTHON} -m pip install aiohttp datasets transformers math-verify" >&2
    exit 1
fi

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        local pgid
        pgid=$(ps -o pgid= -p "${SERVER_PID}" 2>/dev/null | tr -d ' ' || true)
        if [[ -n "${pgid}" && "${pgid}" != "0" ]]; then
            kill -- "-${pgid}" 2>/dev/null || true
        fi
    fi
    pkill -TERM -f "vllm serve ${MODEL}.*--port ${PORT}" 2>/dev/null || true
}
trap stop_server EXIT INT TERM

echo "Starting ${MODEL} on port ${PORT} (TP=${TP})..."
setsid bash -c "VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
    vllm serve '${MODEL}' \
    --port '${PORT}' \
    --dtype bfloat16 \
    --tensor-parallel-size '${TP}' \
    --max-model-len '${MAX_MODEL_LEN}' \
    --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
    --enforce-eager \
    --block-size 64 \
    --max-num-seqs '$(( CONCURRENCY + 1 ))' \
    --no-enable-prefix-caching \
    --no-enable-log-requests \
    --disable-log-stats \
    --trust-remote-code \
    --quantization fp8 \
    --attention-backend FLASH_ATTN \
    --reasoning-parser gemma4 \
    > '${SERVER_LOG}' 2>&1" &
SERVER_PID=$!

echo "Waiting up to 900 seconds for the server..."
for ((attempt = 1; attempt <= 900; attempt++)); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
        echo "Server ready after ${attempt}s."
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server exited during startup. Last log lines:" >&2
        tail -50 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if [[ ${attempt} -eq 900 ]]; then
        echo "Timed out waiting for server. Last log lines:" >&2
        tail -50 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 1
done

export MODEL PORT LIMIT CONCURRENCY MAX_COMPLETION_TOKENS THINKING_BUDGET SEED
export REQUESTS_JSONL SUMMARY_JSON SUMMARY_CSV

"${PYTHON}" - <<'PY'
import asyncio
import csv
import json
import os
import random
import re
import statistics
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer

MODEL = os.environ["MODEL"]
PORT = int(os.environ["PORT"])
LIMIT = int(os.environ["LIMIT"])
CONCURRENCY = int(os.environ["CONCURRENCY"])
MAX_TOKENS = int(os.environ["MAX_COMPLETION_TOKENS"])
THINKING_BUDGET = os.environ.get("THINKING_BUDGET")
SEED = int(os.environ["SEED"])
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

requests_path = Path(os.environ["REQUESTS_JSONL"])
summary_path = Path(os.environ["SUMMARY_JSON"])
csv_path = Path(os.environ["SUMMARY_CSV"])

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
rng = random.Random(SEED)
indices = list(range(len(dataset)))
rng.shuffle(indices)
examples = [dataset[index] for index in indices[:LIMIT]]


def token_count(text):
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def percentile(values, percent):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def score_answer(answer, reference):
    try:
        # MATH-500 answers are raw LaTeX without math delimiters.
        if verify(parse(f"${reference}$"), parse(answer)):
            return True
    except Exception:
        pass

    def normalize(value):
        value = re.sub(r"\s+", "", value).lower()
        value = value.replace("$", "").replace(r"\left", "").replace(r"\right", "")
        return re.sub(r"^[a-z]+\\in", "", value)

    return normalize(reference) in normalize(answer)


async def run_request(session, example_id, example, thinking, order):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": example["problem"]}],
        "chat_template_kwargs": {"enable_thinking": thinking},
        "include_reasoning": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "seed": SEED + example_id,
        "max_completion_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if THINKING_BUDGET and thinking:
        payload["thinking_token_budget"] = int(THINKING_BUDGET)

    reasoning_parts = []
    answer_parts = []
    usage = {}
    first_token_s = None
    first_reasoning_s = None
    first_answer_s = None
    last_reasoning_s = None
    last_answer_s = None
    finish_reason = None
    started = time.perf_counter()

    async with session.post(URL, json=payload) as response:
        if response.status != 200:
            detail = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {detail}")
        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                continue
            chunk = json.loads(body)
            elapsed = time.perf_counter() - started
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta", {})
                reasoning = delta.get("reasoning") or ""
                answer = delta.get("content") or ""
                if (reasoning or answer) and first_token_s is None:
                    first_token_s = elapsed
                if reasoning:
                    first_reasoning_s = first_reasoning_s or elapsed
                    last_reasoning_s = elapsed
                    reasoning_parts.append(reasoning)
                if answer:
                    first_answer_s = first_answer_s or elapsed
                    last_answer_s = elapsed
                    answer_parts.append(answer)

    latency_s = time.perf_counter() - started
    reasoning = "".join(reasoning_parts)
    answer = "".join(answer_parts)
    reasoning_tokens = token_count(reasoning)
    answer_tokens = token_count(answer)
    total_tokens = usage.get("completion_tokens", reasoning_tokens + answer_tokens)
    prompt_tokens = usage.get("prompt_tokens", 0)
    control_tokens = max(0, total_tokens - reasoning_tokens - answer_tokens)
    reasoning_phase_s = (
        max(0.0, last_reasoning_s - first_reasoning_s)
        if first_reasoning_s is not None and last_reasoning_s is not None
        else 0.0
    )
    answer_phase_s = (
        max(0.0, last_answer_s - first_answer_s)
        if first_answer_s is not None and last_answer_s is not None
        else 0.0
    )
    reasoning_tpot_ms = (
        1000 * reasoning_phase_s / (reasoning_tokens - 1)
        if reasoning_tokens > 1
        else 0.0
    )
    answer_tpot_ms = (
        1000 * answer_phase_s / (answer_tokens - 1)
        if answer_tokens > 1
        else 0.0
    )

    return {
        "example_id": example_id,
        "mode": "thinking" if thinking else "non_thinking",
        "pair_order": order,
        "subject": example.get("subject"),
        "level": example.get("level"),
        "problem": example["problem"],
        "reference_answer": example["answer"],
        "answer": answer,
        "reasoning": reasoning,
        "correct": score_answer(answer, example["answer"]),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "latency_s": latency_s,
        "time_to_first_token_s": first_token_s,
        "time_to_first_reasoning_token_s": first_reasoning_s,
        "time_to_first_answer_token_s": first_answer_s,
        "prompt_tokens": prompt_tokens,
        "total_output_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "answer_tokens": answer_tokens,
        "control_tokens": control_tokens,
        "reasoning_phase_s": reasoning_phase_s,
        "answer_phase_s": answer_phase_s,
        "reasoning_tpot_ms": reasoning_tpot_ms,
        "answer_tpot_ms": answer_tpot_ms,
        "output_tokens_per_s": total_tokens / latency_s,
        "total_tokens_per_s": (prompt_tokens + total_tokens) / latency_s,
        "reasoning_tokens_per_s": (
            (reasoning_tokens - 1) / reasoning_phase_s
            if reasoning_tokens > 1 and reasoning_phase_s
            else 0.0
        ),
        "answer_phase_tokens_per_s": (
            (answer_tokens - 1) / answer_phase_s
            if answer_tokens > 1 and answer_phase_s
            else 0.0
        ),
        # Answer efficiency includes prefill and any preceding reasoning time.
        "answer_tokens_per_s": answer_tokens / latency_s,
    }


async def main():
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = []

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        smoke = {"problem": "What is 17 + 25?", "answer": "42"}
        smoke_on = await run_request(session, -1, smoke, True, "smoke")
        smoke_off = await run_request(session, -1, smoke, False, "smoke")
        if not smoke_on["reasoning"]:
            raise RuntimeError(
                "Thinking smoke test returned no reasoning. Check the Gemma 4 "
                "chat template and --reasoning-parser gemma4."
            )
        if smoke_off["reasoning"]:
            raise RuntimeError(
                "Non-thinking smoke test returned reasoning; enable_thinking=False "
                "is not being honored by the loaded chat template."
            )

        measured_started = time.perf_counter()

        async def run_limited(example_id, example, thinking, order):
            async with semaphore:
                result = await run_request(
                    session, example_id, example, thinking, order
                )
                with requests_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(result) + "\n")
                print(
                    f"[{example_id + 1:03d}/{LIMIT}] {result['mode']:<12} "
                    f"correct={int(result['correct'])} "
                    f"tokens={result['total_output_tokens']} "
                    f"think={result['reasoning_tokens']} "
                    f"answer={result['answer_tokens']} "
                    f"finish={result['finish_reason']} "
                    f"latency={result['latency_s']:.2f}s",
                    flush=True,
                )
                return result

        # Alternate pair order to reduce thermal and warm-cache ordering bias.
        for example_id, example in enumerate(examples):
            modes = [False, True] if example_id % 2 == 0 else [True, False]
            for order, thinking in enumerate(modes):
                results.append(
                    await run_limited(example_id, example, thinking, order)
                )
        measured_duration_s = time.perf_counter() - measured_started

    summaries = []
    for mode in ("non_thinking", "thinking"):
        rows = [row for row in results if row["mode"] == mode]
        total_prompt_tokens = sum(row["prompt_tokens"] for row in rows)
        total_tokens = sum(row["total_output_tokens"] for row in rows)
        total_latency_s = sum(row["latency_s"] for row in rows)
        total_reasoning_phase_s = sum(row["reasoning_phase_s"] for row in rows)
        total_answer_phase_s = sum(row["answer_phase_s"] for row in rows)
        total_reasoning_tokens = sum(row["reasoning_tokens"] for row in rows)
        total_answer_tokens = sum(row["answer_tokens"] for row in rows)
        reasoning_decode_intervals = sum(
            max(0, row["reasoning_tokens"] - 1) for row in rows
        )
        answer_decode_intervals = sum(
            max(0, row["answer_tokens"] - 1) for row in rows
        )
        reasoning_ttfts = [
            row["time_to_first_reasoning_token_s"]
            for row in rows
            if row["time_to_first_reasoning_token_s"] is not None
        ]
        answer_ttfts = [
            row["time_to_first_answer_token_s"]
            for row in rows
            if row["time_to_first_answer_token_s"] is not None
        ]
        reasoning_tpots = [
            row["reasoning_tpot_ms"]
            for row in rows
            if row["reasoning_tokens"] > 1
        ]
        answer_tpots = [
            row["answer_tpot_ms"] for row in rows if row["answer_tokens"] > 1
        ]
        correct = sum(row["correct"] for row in rows)
        summary = {
            "mode": mode,
            "samples": len(rows),
            "accuracy": correct / len(rows),
            "correct": correct,
            "truncated": sum(row["truncated"] for row in rows),
            "truncation_rate": statistics.fmean(row["truncated"] for row in rows),
            "mean_latency_s": statistics.fmean(row["latency_s"] for row in rows),
            "p50_latency_s": percentile([row["latency_s"] for row in rows], 50),
            "p95_latency_s": percentile([row["latency_s"] for row in rows], 95),
            "request_throughput_req_s": len(rows) / total_latency_s,
            "output_throughput_tokens_s": total_tokens / total_latency_s,
            "total_token_throughput_tokens_s": (
                total_prompt_tokens + total_tokens
            ) / total_latency_s,
            "reasoning_throughput_tokens_s": (
                reasoning_decode_intervals / total_reasoning_phase_s
                if total_reasoning_phase_s
                else 0.0
            ),
            "answer_phase_throughput_tokens_s": (
                answer_decode_intervals / total_answer_phase_s
                if total_answer_phase_s
                else 0.0
            ),
            "answer_e2e_throughput_tokens_s": total_answer_tokens / total_latency_s,
            "mean_reasoning_ttft_ms": (
                1000 * statistics.fmean(reasoning_ttfts)
                if reasoning_ttfts
                else None
            ),
            "p50_reasoning_ttft_ms": (
                1000 * percentile(reasoning_ttfts, 50) if reasoning_ttfts else None
            ),
            "p95_reasoning_ttft_ms": (
                1000 * percentile(reasoning_ttfts, 95) if reasoning_ttfts else None
            ),
            "mean_answer_ttft_ms": (
                1000 * statistics.fmean(answer_ttfts) if answer_ttfts else None
            ),
            "p50_answer_ttft_ms": (
                1000 * percentile(answer_ttfts, 50) if answer_ttfts else None
            ),
            "p95_answer_ttft_ms": (
                1000 * percentile(answer_ttfts, 95) if answer_ttfts else None
            ),
            "mean_reasoning_tpot_ms": (
                statistics.fmean(reasoning_tpots) if reasoning_tpots else None
            ),
            "p50_reasoning_tpot_ms": (
                percentile(reasoning_tpots, 50) if reasoning_tpots else None
            ),
            "p95_reasoning_tpot_ms": (
                percentile(reasoning_tpots, 95) if reasoning_tpots else None
            ),
            "mean_answer_tpot_ms": (
                statistics.fmean(answer_tpots) if answer_tpots else None
            ),
            "p50_answer_tpot_ms": (
                percentile(answer_tpots, 50) if answer_tpots else None
            ),
            "p95_answer_tpot_ms": (
                percentile(answer_tpots, 95) if answer_tpots else None
            ),
            "mean_time_to_first_answer_s": statistics.fmean(
                row["time_to_first_answer_token_s"] or row["latency_s"]
                for row in rows
            ),
            "total_output_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "total_answer_tokens": total_answer_tokens,
            "mean_output_tokens": statistics.fmean(
                row["total_output_tokens"] for row in rows
            ),
            "mean_reasoning_tokens": statistics.fmean(
                row["reasoning_tokens"] for row in rows
            ),
            "mean_answer_tokens": statistics.fmean(
                row["answer_tokens"] for row in rows
            ),
            "accuracy_per_1k_output_tokens": (
                1000 * correct / total_tokens if total_tokens else 0.0
            ),
        }
        summaries.append(summary)

    paired = []
    for example_id in range(len(examples)):
        off = next(
            row
            for row in results
            if row["example_id"] == example_id and row["mode"] == "non_thinking"
        )
        on = next(
            row
            for row in results
            if row["example_id"] == example_id and row["mode"] == "thinking"
        )
        paired.append(
            {
                "example_id": example_id,
                "accuracy_delta": int(on["correct"]) - int(off["correct"]),
                "latency_ratio": on["latency_s"] / off["latency_s"],
                "output_token_ratio": on["total_output_tokens"]
                / max(1, off["total_output_tokens"]),
            }
        )

    report = {
        "model": MODEL,
        "dataset": "HuggingFaceH4/MATH-500:test",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "max_completion_tokens": MAX_TOKENS,
        "thinking_budget": int(THINKING_BUDGET) if THINKING_BUDGET else None,
        "seed": SEED,
        "concurrency": CONCURRENCY,
        "measured_duration_s": measured_duration_s,
        "overall_request_throughput_req_s": len(results) / measured_duration_s,
        "overall_output_throughput_tokens_s": (
            sum(row["total_output_tokens"] for row in results)
            / measured_duration_s
        ),
        "overall_total_token_throughput_tokens_s": (
            sum(
                row["prompt_tokens"] + row["total_output_tokens"]
                for row in results
            )
            / measured_duration_s
        ),
        "summaries": summaries,
        "paired_mean_accuracy_delta": statistics.fmean(
            row["accuracy_delta"] for row in paired
        ),
        "paired_median_latency_ratio": statistics.median(
            row["latency_ratio"] for row in paired
        ),
        "paired_median_output_token_ratio": statistics.median(
            row["output_token_ratio"] for row in paired
        ),
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print("\n" + json.dumps(report, indent=2))


asyncio.run(main())
PY

echo "Request details: ${REQUESTS_JSONL}"
echo "Summary JSON:   ${SUMMARY_JSON}"
echo "Summary CSV:    ${SUMMARY_CSV}"