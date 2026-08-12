#!/bin/bash
# Paired thinking-mode benchmark for google/gemma-4-31B-it on Intel XPU.
#
# Compares Gemma 4 thinking ON and OFF on paired task prompts and seeds.
# Reports answer accuracy, latency, total/reasoning/answer token counts, and
# throughput. The server's usage count is exact for total generated tokens;
# reasoning and answer text are re-tokenized for attribution, with the residual
# reported as control tokens (channel delimiters and other special tokens).
#
# Usage:
#   bash bench_gemma4_thinking.sh
#   bash bench_gemma4_thinking.sh --dataset gsm8k
#   LIMIT=100 TP=2 MAX_COMPLETION_TOKENS=8192 bash bench_gemma4_thinking.sh
#   THINKING_BUDGET=4096 bash bench_gemma4_thinking.sh
#   RUN_DIR=/tmp/gemma4_thinking/math500/<timestamp> bash bench_gemma4_thinking.sh
#
# Runs directly inside the vLLM container. The model must already be accessible
# from Hugging Face. Override PYTHON only when the container uses another path.
set -euo pipefail

DATASET="${DATASET:-math500}"

usage() {
    cat <<'EOF'
Usage: bash bench_gemma4_thinking.sh [--dataset NAME]

Datasets:
  math500    HuggingFaceH4/MATH-500 (default)
  gsm8k      openai/gsm8k
  aime       AI-MO/aimo-validation-aime
  gpqa       Idavidrein/gpqa (gpqa_diamond)
    humaneval  openai/openai_humaneval (code reasoning; alias: code)
    mrcr       openai/mrcr (long-context reasoning; alias: long_context)

The DATASET environment variable is also supported. --dataset takes precedence.
EOF
}

while (( $# )); do
    case "$1" in
        --dataset)
            if (( $# < 2 )); then
                echo "--dataset requires a value" >&2
                usage >&2
                exit 2
            fi
            DATASET="$2"
            shift 2
            ;;
        --dataset=*)
            DATASET="${1#*=}"
            shift
            ;;
        --list-datasets)
            usage
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${DATASET}" in
    code) DATASET="humaneval" ;;
    long-context|long_context) DATASET="mrcr" ;;
esac

case "${DATASET}" in
    math500|gsm8k|aime|gpqa|humaneval|mrcr) ;;
    *)
        echo "Unsupported dataset: ${DATASET}" >&2
        usage >&2
        exit 2
        ;;
esac

MODEL="${MODEL:-google/gemma-4-31B-it}"
PORT="${PORT:-8260}"
TP="${TP:-2}"
LIMIT="${LIMIT:-50}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
THINKING_BUDGET="${THINKING_BUDGET:-}"
SEED="${SEED:-42}"
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_BACKOFF_S="${RETRY_BACKOFF_S:-2}"
CODE_TIMEOUT_S="${CODE_TIMEOUT_S:-10}"
MRCR_NEEDLES="${MRCR_NEEDLES:-2,4,8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
PYTHON="${PYTHON:-python3}"
RESULT_DIR="${RESULT_DIR:-/tmp/gemma4_thinking}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${RESULT_DIR}/${DATASET}/${TIMESTAMP}}"
SERVER_LOG="${RUN_DIR}/server.log"
REQUESTS_JSONL="${RUN_DIR}/requests.jsonl"
FAILURES_JSONL="${RUN_DIR}/failures.jsonl"
RUN_STATE_JSON="${RUN_DIR}/run_state.json"
SUMMARY_JSON="${RUN_DIR}/summary.json"
SUMMARY_CSV="${RUN_DIR}/summary.csv"

mkdir -p "${RUN_DIR}"

if [[ "${CONCURRENCY}" != "1" ]]; then
    echo "CONCURRENCY=${CONCURRENCY} is not supported by this paired sequential benchmark." >&2
    echo "Use CONCURRENCY=1; run a separate load benchmark for concurrent throughput." >&2
    exit 1
fi

if (( MAX_COMPLETION_TOKENS >= MAX_MODEL_LEN )); then
    echo "MAX_COMPLETION_TOKENS (${MAX_COMPLETION_TOKENS}) must be smaller than MAX_MODEL_LEN (${MAX_MODEL_LEN})." >&2
    echo "Every request also consumes prompt tokens: prompt_tokens + MAX_COMPLETION_TOKENS <= MAX_MODEL_LEN." >&2
    echo "Increase MAX_MODEL_LEN instead, e.g. MAX_MODEL_LEN=24576 MAX_COMPLETION_TOKENS=16384." >&2
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

echo "Starting ${MODEL} on port ${PORT} (TP=${TP}, dataset=${DATASET})..."
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
    >> '${SERVER_LOG}' 2>&1" &
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

export DATASET MODEL PORT TP LIMIT CONCURRENCY MAX_COMPLETION_TOKENS MAX_MODEL_LEN
export THINKING_BUDGET SEED GPU_MEMORY_UTILIZATION
export MAX_RETRIES RETRY_BACKOFF_S CODE_TIMEOUT_S MRCR_NEEDLES
export REQUESTS_JSONL FAILURES_JSONL RUN_STATE_JSON SUMMARY_JSON SUMMARY_CSV

"${PYTHON}" - <<'PY'
import asyncio
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

import aiohttp
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer

MODEL = os.environ["MODEL"]
DATASET = os.environ["DATASET"]
PORT = int(os.environ["PORT"])
TP = int(os.environ["TP"])
LIMIT = int(os.environ["LIMIT"])
CONCURRENCY = int(os.environ["CONCURRENCY"])
MAX_TOKENS = int(os.environ["MAX_COMPLETION_TOKENS"])
MAX_MODEL_LEN = int(os.environ["MAX_MODEL_LEN"])
THINKING_BUDGET = os.environ.get("THINKING_BUDGET")
SEED = int(os.environ["SEED"])
GPU_MEMORY_UTILIZATION = float(os.environ["GPU_MEMORY_UTILIZATION"])
MAX_RETRIES = int(os.environ["MAX_RETRIES"])
RETRY_BACKOFF_S = float(os.environ["RETRY_BACKOFF_S"])
CODE_TIMEOUT_S = float(os.environ["CODE_TIMEOUT_S"])
MRCR_NEEDLES = [
    int(value) for value in os.environ["MRCR_NEEDLES"].split(",") if value
]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

requests_path = Path(os.environ["REQUESTS_JSONL"])
failures_path = Path(os.environ["FAILURES_JSONL"])
state_path = Path(os.environ["RUN_STATE_JSON"])
summary_path = Path(os.environ["SUMMARY_JSON"])
csv_path = Path(os.environ["SUMMARY_CSV"])

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

if LIMIT <= 0:
    raise ValueError("LIMIT must be greater than zero")

DATASET_SPECS = {
    "math500": {
        "source": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "scorer": "math_verify_final_answer_v2",
    },
    "gsm8k": {
        "source": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "scorer": "last_number_exact_v1",
    },
    "aime": {
        "source": "AI-MO/aimo-validation-aime",
        "config": None,
        "split": "train",
        "scorer": "math_verify_final_answer_v2",
    },
    "gpqa": {
        "source": "Idavidrein/gpqa",
        "config": "gpqa_diamond",
        "split": "train",
        "scorer": "multiple_choice_exact_v1",
    },
    "humaneval": {
        "source": "openai/openai_humaneval",
        "config": None,
        "split": "test",
        "scorer": "execution_tests_v1",
    },
    "mrcr": {
        "source": "openai/mrcr",
        "config": None,
        "split": "train",
        "scorer": "prefix_sequence_match_v1",
    },
}


def extract_boxed(text):
    values = []
    marker = r"\boxed{"
    offset = 0
    while True:
        start = text.find(marker, offset)
        if start < 0:
            break
        content_start = start + len(marker)
        depth = 1
        position = content_start
        while position < len(text) and depth:
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
            position += 1
        if depth == 0:
            values.append(text[content_start : position - 1].strip())
            offset = position
        else:
            offset = content_start
    return values[-1] if values else None


def extract_final_math_answer(text):
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    matches = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip().rstrip(".")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].rstrip(".") if lines else ""


def sample_rows(hf_dataset):
    if LIMIT > len(hf_dataset):
        raise ValueError(
            f"LIMIT={LIMIT} exceeds {DATASET} size ({len(hf_dataset)})"
        )
    indices = list(range(len(hf_dataset)))
    random.Random(SEED).shuffle(indices)
    selected = indices[:LIMIT]
    return selected, [(index, hf_dataset[index]) for index in selected]


def load_standard_examples(spec):
    kwargs = {"split": spec["split"]}
    if spec["config"]:
        hf_dataset = load_dataset(spec["source"], spec["config"], **kwargs)
    else:
        hf_dataset = load_dataset(spec["source"], **kwargs)
    selected, rows = sample_rows(hf_dataset)
    normalized = []
    for source_index, row in rows:
        metadata = {"source_index": source_index}
        if DATASET == "math500":
            problem = row["problem"]
            reference = row["answer"]
            metadata.update(subject=row.get("subject"), level=row.get("level"))
            prompt = problem
        elif DATASET == "gsm8k":
            problem = row["question"]
            reference = row["answer"].split("####")[-1].strip()
            prompt = (
                f"{problem}\n\nReason step by step. End with "
                "'Final answer: <number>'."
            )
        elif DATASET == "aime":
            problem = row["problem"]
            solution = row["solution"]
            reference = extract_boxed(solution) or extract_final_math_answer(solution)
            metadata["reference_solution"] = solution
            prompt = (
                f"{problem}\n\nReason step by step and put only the final answer "
                "inside \\boxed{}."
            )
        elif DATASET == "gpqa":
            problem = row["Question"]
            choices = [
                (row["Correct Answer"], True),
                (row["Incorrect Answer 1"], False),
                (row["Incorrect Answer 2"], False),
                (row["Incorrect Answer 3"], False),
            ]
            random.Random(SEED + source_index).shuffle(choices)
            letters = "ABCD"
            reference = next(
                letters[index]
                for index, (_, is_correct) in enumerate(choices)
                if is_correct
            )
            option_text = "\n".join(
                f"{letters[index]}. {choice}"
                for index, (choice, _) in enumerate(choices)
            )
            prompt = (
                f"{problem}\n\n{option_text}\n\nReason through the question. "
                "End with 'Final answer: <letter>'."
            )
            metadata["choices"] = [choice for choice, _ in choices]
        elif DATASET == "humaneval":
            problem = row["prompt"]
            reference = row["canonical_solution"]
            prompt = (
                "Complete the following Python function. Return executable Python "
                "code only, without Markdown fences.\n\n" + problem
            )
            metadata.update(
                task_id=row["task_id"],
                test=row["test"],
                entry_point=row["entry_point"],
                function_prompt=row["prompt"],
            )
        else:
            raise AssertionError(f"Unhandled dataset adapter: {DATASET}")
        normalized.append(
            {
                "problem": problem,
                "messages": [{"role": "user", "content": prompt}],
                "answer": reference,
                "metadata": metadata,
            }
        )
    return selected, normalized


def load_mrcr_examples(spec):
    shards = {
        2: "2needle/2needle_0.parquet",
        4: "4needle/4needle_0.parquet",
        8: "8needle/8needle_0.parquet",
    }
    if not MRCR_NEEDLES or any(value not in shards for value in MRCR_NEEDLES):
        raise ValueError("MRCR_NEEDLES must contain values from: 2,4,8")
    max_prompt_tokens = MAX_MODEL_LEN - MAX_TOKENS - 256
    if max_prompt_tokens < 512:
        raise ValueError(
            "MRCR requires MAX_MODEL_LEN - MAX_COMPLETION_TOKENS >= 768"
        )
    per_bucket, leftover = divmod(LIMIT, len(MRCR_NEEDLES))
    selected = []
    normalized = []
    for bucket_index, needle_count in enumerate(MRCR_NEEDLES):
        target = per_bucket + (1 if bucket_index < leftover else 0)
        if target == 0:
            continue
        stream = load_dataset(
            spec["source"],
            data_files=shards[needle_count],
            split=spec["split"],
            streaming=True,
        ).shuffle(seed=SEED + needle_count, buffer_size=16)
        taken = 0
        for source_index, row in enumerate(stream):
            if int(row.get("n_chars", 0)) > max_prompt_tokens * 4:
                continue
            messages = row["prompt"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            prompt_tokens = len(
                tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                )
            )
            if prompt_tokens > max_prompt_tokens:
                continue
            selected.append(f"{needle_count}:{source_index}")
            normalized.append(
                {
                    "problem": f"MRCR {needle_count}-needle retrieval",
                    "messages": list(messages),
                    "answer": row["answer"],
                    "metadata": {
                        "source_index": source_index,
                        "n_needles": needle_count,
                        "prompt_tokens_preflight": prompt_tokens,
                        "random_string_to_prepend": row[
                            "random_string_to_prepend"
                        ],
                    },
                }
            )
            taken += 1
            if taken >= target:
                break
        if taken < target:
            raise RuntimeError(
                f"Only {taken}/{target} MRCR samples fit for {needle_count} needles; "
                "increase MAX_MODEL_LEN or reduce MAX_COMPLETION_TOKENS"
            )
    return selected, normalized


dataset_spec = DATASET_SPECS[DATASET]
if DATASET == "mrcr":
    selected_indices, examples = load_mrcr_examples(dataset_spec)
else:
    selected_indices, examples = load_standard_examples(dataset_spec)

run_config = {
    "model": MODEL,
    "dataset_key": DATASET,
    "dataset": (
        f"{dataset_spec['source']}"
        f"{':' + dataset_spec['config'] if dataset_spec['config'] else ''}"
        f":{dataset_spec['split']}"
    ),
    "scorer": dataset_spec["scorer"],
    "selected_indices": selected_indices,
    "tensor_parallel_size": TP,
    "max_model_len": MAX_MODEL_LEN,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    "max_completion_tokens": MAX_TOKENS,
    "thinking_budget": int(THINKING_BUDGET) if THINKING_BUDGET else None,
    "code_timeout_s": CODE_TIMEOUT_S if DATASET == "humaneval" else None,
    "mrcr_needles": MRCR_NEEDLES if DATASET == "mrcr" else None,
    "seed": SEED,
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
}


def write_json_atomic(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("config") != run_config:
        raise RuntimeError(
            f"Cannot resume {state_path.parent}: benchmark configuration changed"
        )
    active_duration_s = float(state.get("active_duration_s", 0.0))
elif requests_path.exists() and requests_path.stat().st_size:
    raise RuntimeError(
        f"Cannot safely resume {requests_path}: {state_path.name} is missing"
    )
else:
    active_duration_s = 0.0
    write_json_atomic(
        state_path,
        {"config": run_config, "active_duration_s": active_duration_s},
    )

checkpoint_results = {}
if requests_path.exists():
    with requests_path.open(encoding="utf-8") as checkpoint:
        for line_number, line in enumerate(checkpoint, start=1):
            try:
                row = json.loads(line)
                key = (int(row["example_id"]), row["mode"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"Invalid checkpoint at {requests_path}:{line_number}"
                ) from error
            checkpoint_results[key] = row


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


def score_math(answer, reference):
    extracted = extract_final_math_answer(answer)
    try:
        parsed_reference = parse(f"${reference}$")
    except Exception as error:
        return {
            "correct": None,
            "score_valid": False,
            "score_status": "reference_parse_error",
            "score_error": str(error),
            "extracted_answer": extracted,
            "task_score": None,
        }
    if not parsed_reference:
        return {
            "correct": None,
            "score_valid": False,
            "score_status": "reference_not_parsed",
            "score_error": None,
            "extracted_answer": extracted,
            "task_score": None,
        }

    try:
        parsed_answer = parse(f"${extracted}$")
    except Exception as error:
        return {
            "correct": False,
            "score_valid": True,
            "score_status": "answer_parse_error",
            "score_error": str(error),
            "extracted_answer": extracted,
            "task_score": 0.0,
        }
    if not parsed_answer:
        return {
            "correct": False,
            "score_valid": True,
            "score_status": "answer_not_parsed",
            "score_error": None,
            "extracted_answer": extracted,
            "task_score": 0.0,
        }

    try:
        correct = bool(verify(parsed_reference, parsed_answer))
    except Exception as error:
        return {
            "correct": None,
            "score_valid": False,
            "score_status": "verification_error",
            "score_error": str(error),
            "extracted_answer": extracted,
            "task_score": None,
        }
    return {
        "correct": correct,
        "score_valid": True,
        "score_status": "verified",
        "score_error": None,
        "extracted_answer": extracted,
        "task_score": float(correct),
    }


def extract_last_number(text):
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else ""


def score_gsm8k(answer, reference):
    extracted = extract_last_number(extract_final_math_answer(answer))
    try:
        correct = Decimal(extracted) == Decimal(reference.replace(",", ""))
    except InvalidOperation:
        correct = False
    return {
        "correct": correct,
        "score_valid": True,
        "score_status": "numeric_exact" if extracted else "answer_not_parsed",
        "score_error": None,
        "extracted_answer": extracted,
        "task_score": float(correct),
    }


def score_gpqa(answer, reference):
    matches = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\(?([A-D])\)?\b",
        answer,
        flags=re.IGNORECASE,
    )
    extracted = matches[-1].upper() if matches else ""
    correct = extracted == reference
    return {
        "correct": correct,
        "score_valid": True,
        "score_status": "choice_exact" if extracted else "answer_not_parsed",
        "score_error": None,
        "extracted_answer": extracted,
        "task_score": float(correct),
    }


def extract_python_code(answer):
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", answer, flags=re.DOTALL)
    return (fenced[-1] if fenced else answer).strip()


def score_humaneval(answer, example):
    metadata = example["metadata"]
    code = extract_python_code(answer)
    if re.search(rf"^\s*def\s+{re.escape(metadata['entry_point'])}\s*\(", code, re.M):
        candidate = code
    else:
        candidate = metadata["function_prompt"] + code
    program = (
        candidate
        + "\n\n"
        + metadata["test"]
        + f"\n\ncheck({metadata['entry_point']})\n"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="gemma4_humaneval_") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=directory,
                text=True,
                capture_output=True,
                timeout=CODE_TIMEOUT_S,
                check=False,
            )
        correct = completed.returncode == 0
        error = None if correct else (completed.stderr or completed.stdout)[-2000:]
        status = "tests_passed" if correct else "tests_failed"
    except subprocess.TimeoutExpired:
        correct = False
        error = f"Execution exceeded {CODE_TIMEOUT_S:g}s"
        status = "execution_timeout"
    except Exception as exception:
        return {
            "correct": None,
            "score_valid": False,
            "score_status": "execution_error",
            "score_error": str(exception),
            "extracted_answer": code,
            "task_score": None,
        }
    return {
        "correct": correct,
        "score_valid": True,
        "score_status": status,
        "score_error": error,
        "extracted_answer": code,
        "task_score": float(correct),
    }


def score_mrcr(answer, example):
    prefix = example["metadata"]["random_string_to_prepend"]
    if not answer.startswith(prefix):
        ratio = 0.0
        status = "prefix_missing"
    else:
        ratio = SequenceMatcher(
            a=example["answer"], b=answer[len(prefix) :], autojunk=False
        ).ratio()
        status = "prefix_matched"
    return {
        "correct": ratio == 1.0,
        "score_valid": True,
        "score_status": status,
        "score_error": None,
        "extracted_answer": answer[len(prefix) :] if answer.startswith(prefix) else "",
        "task_score": ratio,
    }


def score_answer(answer, example):
    if DATASET in {"math500", "aime"}:
        return score_math(answer, example["answer"])
    if DATASET == "gsm8k":
        return score_gsm8k(answer, example["answer"])
    if DATASET == "gpqa":
        return score_gpqa(answer, example["answer"])
    if DATASET == "humaneval":
        return score_humaneval(answer, example)
    if DATASET == "mrcr":
        return score_mrcr(answer, example)
    raise AssertionError(f"Unhandled scorer: {DATASET}")


async def run_request(session, example_id, example, thinking, order, score=True):
    payload = {
        "model": MODEL,
        "messages": example["messages"],
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
    score_result = score_answer(answer, example) if score else {}

    return {
        "dataset": DATASET,
        "example_id": example_id,
        "mode": "thinking" if thinking else "non_thinking",
        "pair_order": order,
        "source_index": example.get("metadata", {}).get("source_index"),
        "subject": example.get("metadata", {}).get("subject"),
        "level": example.get("metadata", {}).get("level"),
        "task_metadata": example.get("metadata", {}),
        "problem": example["problem"],
        "messages": example["messages"],
        "reference_answer": example["answer"],
        "answer": answer,
        "reasoning": reasoning,
        **score_result,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "missing_answer": first_answer_s is None,
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
    global active_duration_s

    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results_by_key = dict(checkpoint_results)

    class RequestFailed(RuntimeError):
        def __init__(self, errors):
            super().__init__(errors[-1])
            self.errors = errors

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def run_with_retries(
            example_id, example, thinking, order, score=True
        ):
            errors = []
            for attempt in range(1, MAX_RETRIES + 2):
                try:
                    result = await run_request(
                        session, example_id, example, thinking, order, score
                    )
                    result["attempts"] = attempt
                    return result
                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError,
                        json.JSONDecodeError, UnicodeDecodeError) as error:
                    errors.append(f"{type(error).__name__}: {error}")
                    if attempt <= MAX_RETRIES:
                        await asyncio.sleep(RETRY_BACKOFF_S * attempt)
            raise RequestFailed(errors)

        smoke = {
            "problem": "What is 17 + 25?",
            "messages": [{"role": "user", "content": "What is 17 + 25?"}],
            "answer": "42",
            "metadata": {},
        }
        smoke_on = await run_with_retries(-1, smoke, True, "smoke", False)
        smoke_off = await run_with_retries(-1, smoke, False, "smoke", False)
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

        async def run_limited(example_id, example, thinking, order):
            global active_duration_s

            mode = "thinking" if thinking else "non_thinking"
            key = (example_id, mode)
            if key in results_by_key:
                print(
                    f"[{example_id + 1:03d}/{LIMIT}] {mode:<12} resumed",
                    flush=True,
                )
                return results_by_key[key]

            async with semaphore:
                request_started = time.perf_counter()
                try:
                    result = await run_with_retries(
                        example_id, example, thinking, order
                    )
                except RequestFailed as error:
                    active_duration_s += time.perf_counter() - request_started
                    failure = {
                        "example_id": example_id,
                        "mode": mode,
                        "pair_order": order,
                        "attempts": MAX_RETRIES + 1,
                        "errors": error.errors,
                        "recorded_at_unix_s": time.time(),
                    }
                    write_json_atomic(
                        state_path,
                        {
                            "config": run_config,
                            "active_duration_s": active_duration_s,
                        },
                    )
                    with failures_path.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(failure) + "\n")
                    print(
                        f"[{example_id + 1:03d}/{LIMIT}] {mode:<12} FAILED "
                        f"after {MAX_RETRIES + 1} attempts: {error}",
                        flush=True,
                    )
                    return None

                active_duration_s += time.perf_counter() - request_started
                write_json_atomic(
                    state_path,
                    {
                        "config": run_config,
                        "active_duration_s": active_duration_s,
                    },
                )
                with requests_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(result) + "\n")
                results_by_key[key] = result
                print(
                    f"[{example_id + 1:03d}/{LIMIT}] {result['mode']:<12} "
                    f"correct={result['correct']} "
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
                await run_limited(example_id, example, thinking, order)

    results = list(results_by_key.values())
    measured_duration_s = active_duration_s

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
        scored_rows = [row for row in rows if row.get("score_valid", True)]
        task_scores = [
            row["task_score"]
            for row in scored_rows
            if row.get("task_score") is not None
        ]
        correct = sum(row["correct"] is True for row in scored_rows)
        missing_answers = sum(
            row.get("missing_answer", row["answer_tokens"] == 0) for row in rows
        )
        summary = {
            "mode": mode,
            "samples": len(rows),
            "attempted_samples": LIMIT,
            "failed_requests": LIMIT - len(rows),
            "scored_samples": len(scored_rows),
            "scoring_errors": len(rows) - len(scored_rows),
            "accuracy": correct / len(scored_rows) if scored_rows else None,
            "mean_task_score": (
                statistics.fmean(task_scores) if task_scores else None
            ),
            "correct": correct,
            "missing_answers": missing_answers,
            "missing_answer_rate": missing_answers / len(rows) if rows else None,
            "truncated": sum(row["truncated"] for row in rows),
            "truncation_rate": (
                statistics.fmean(row["truncated"] for row in rows)
                if rows
                else None
            ),
            "mean_latency_s": (
                statistics.fmean(row["latency_s"] for row in rows)
                if rows
                else None
            ),
            "p50_latency_s": (
                percentile([row["latency_s"] for row in rows], 50)
                if rows
                else None
            ),
            "p95_latency_s": (
                percentile([row["latency_s"] for row in rows], 95)
                if rows
                else None
            ),
            "request_throughput_req_s": (
                len(rows) / total_latency_s if total_latency_s else 0.0
            ),
            "output_throughput_tokens_s": (
                total_tokens / total_latency_s if total_latency_s else 0.0
            ),
            "total_token_throughput_tokens_s": (
                (total_prompt_tokens + total_tokens) / total_latency_s
                if total_latency_s
                else 0.0
            ),
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
            "answer_e2e_throughput_tokens_s": (
                total_answer_tokens / total_latency_s if total_latency_s else 0.0
            ),
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
            "mean_time_to_first_answer_s": (
                statistics.fmean(answer_ttfts) if answer_ttfts else None
            ),
            "total_output_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "total_answer_tokens": total_answer_tokens,
            "mean_output_tokens": statistics.fmean(
                row["total_output_tokens"] for row in rows
            ) if rows else None,
            "mean_reasoning_tokens": statistics.fmean(
                row["reasoning_tokens"] for row in rows
            ) if rows else None,
            "mean_answer_tokens": statistics.fmean(
                row["answer_tokens"] for row in rows
            ) if rows else None,
            "accuracy_per_1k_output_tokens": (
                1000 * correct / total_tokens if total_tokens else 0.0
            ),
        }
        summaries.append(summary)

    paired = []
    for example_id in range(len(examples)):
        off = results_by_key.get((example_id, "non_thinking"))
        on = results_by_key.get((example_id, "thinking"))
        if off is None or on is None:
            continue
        paired.append(
            {
                "example_id": example_id,
                "accuracy_delta": (
                    int(on["correct"]) - int(off["correct"])
                    if on.get("score_valid", True)
                    and off.get("score_valid", True)
                    else None
                ),
                "task_score_delta": (
                    on["task_score"] - off["task_score"]
                    if on.get("task_score") is not None
                    and off.get("task_score") is not None
                    else None
                ),
                "latency_ratio": on["latency_s"] / off["latency_s"],
                "output_token_ratio": on["total_output_tokens"]
                / max(1, off["total_output_tokens"]),
            }
        )

    report = {
        "model": MODEL,
        "dataset_key": DATASET,
        "dataset": run_config["dataset"],
        "scorer": dataset_spec["scorer"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "max_completion_tokens": MAX_TOKENS,
        "thinking_budget": int(THINKING_BUDGET) if THINKING_BUDGET else None,
        "seed": SEED,
        "concurrency": CONCURRENCY,
        "measured_duration_s": measured_duration_s,
        "successful_requests": len(results),
        "failed_requests": 2 * LIMIT - len(results),
        "complete_pairs": len(paired),
        "overall_request_throughput_req_s": (
            len(results) / measured_duration_s if measured_duration_s else 0.0
        ),
        "overall_output_throughput_tokens_s": (
            sum(row["total_output_tokens"] for row in results)
            / measured_duration_s
            if measured_duration_s
            else 0.0
        ),
        "overall_total_token_throughput_tokens_s": (
            sum(
                row["prompt_tokens"] + row["total_output_tokens"]
                for row in results
            )
            / measured_duration_s
            if measured_duration_s
            else 0.0
        ),
        "summaries": summaries,
        "paired_scorable_samples": sum(
            row["accuracy_delta"] is not None for row in paired
        ),
        "paired_mean_accuracy_delta": (
            statistics.fmean(
                row["accuracy_delta"]
                for row in paired
                if row["accuracy_delta"] is not None
            )
            if any(row["accuracy_delta"] is not None for row in paired)
            else None
        ),
        "paired_mean_task_score_delta": (
            statistics.fmean(
                row["task_score_delta"]
                for row in paired
                if row["task_score_delta"] is not None
            )
            if any(row["task_score_delta"] is not None for row in paired)
            else None
        ),
        "paired_median_latency_ratio": (
            statistics.median(row["latency_ratio"] for row in paired)
            if paired
            else None
        ),
        "paired_median_output_token_ratio": (
            statistics.median(row["output_token_ratio"] for row in paired)
            if paired
            else None
        ),
    }
    write_json_atomic(summary_path, report)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print("\n" + json.dumps(report, indent=2))


asyncio.run(main())
PY

echo "Request details: ${REQUESTS_JSONL}"
echo "Request failures: ${FAILURES_JSONL}"
echo "Run state:       ${RUN_STATE_JSON}"
echo "Summary JSON:   ${SUMMARY_JSON}"
echo "Summary CSV:    ${SUMMARY_CSV}"