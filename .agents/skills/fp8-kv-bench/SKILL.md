---
name: fp8-kv-bench
description: >
  FP8 KV cache benchmarking on Intel Arc Pro B70 XPU workstation using vLLM.
  Use when: running fp8 kv benchmark, repeating bench with different models,
  checking benchmark progress or results, diagnosing empty CSV or failed run,
  exporting results from the B70, adding a new model to the benchmark,
  running accuracy perf sla or long-context benchmark, monitoring ruler eval,
  backfilling csv from results json, understanding benchmark output files.
---

# FP8 KV Cache Benchmark Skill

## Environment Facts

| Item | Value |
|---|---|
| Container name | `vllm-test` |
| Container image | `vllm-xpu-kernel-0.1.9:latest` |
| Scripts directory | `/home/intel/vllm/` |
| Results inside container | per-script: `/tmp/fp8kv_perf/`, `/tmp/fp8kv_sla/`, `/tmp/fp8kv_long_context/`, `/tmp/fp8kv_accuracy/`, `/tmp/fp8_kv_sweep/` |
| Results on host (via /tmp mount) | same names under `~/LLM/` (e.g. `~/LLM/fp8kv_perf/`) |
| XPU kernel cache (host) | `/home/intel/LLM/xpu_cache/` (bind-mounted to `/root/.cache` in container) |
| vLLM version | v0.20.1rc1.dev on XPU |
| Hardware | 4× Intel Arc Pro B70, 32 GiB GDDR6 each (cards 0–3) |

**Critical**: Scripts run **as root directly inside the container** — NOT via `docker exec` from the host.
The `/tmp` inside the container is bind-mounted to `/home/intel/LLM` on the host.
All scripts use `setsid bash -c "..."` to launch the vLLM server in the background.

### Starting the container

```bash
mkdir -p /home/intel/LLM/xpu_cache

docker run --rm -td --privileged --network=host --ipc=host \
  -e http_proxy=http://proxy-dmz.intel.com:912 \
  -e https_proxy=http://proxy-dmz.intel.com:912 \
  -e no_proxy=10.0.0.0/8,habana-labs.com,.habana-labs.com,intel.com,.intel.com,127.0.0.1,localhost \
  -e HF_HOME=/tmp \
  -e HF_TOKEN=<your-hf-token> \
  -v ~/LLM:/tmp \
  -v /home/intel/LLM/xpu_cache:/root/.cache \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v $(pwd):$(pwd) -w $(pwd) \
  --name=vllm-test \
  --device /dev/dri:/dev/dri \
  --entrypoint=/bin/bash \
  vllm-xpu-kernel-0.1.9:latest
```

> **Note:** The `xpu_cache` bind mount persists the XPU/IGC kernel compilation cache (~125 GB on first run) across container restarts. Without it, every new container pays the full compilation cost.

## Model Registry

| Shortname | Full HF Model ID | TP | Extra flags |
|---|---|---|---|
| `llama31` | `meta-llama/Llama-3.1-8B-Instruct` | 1 | `--quantization fp8` |
| `deepseekr1` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 1 | `--trust-remote-code --quantization fp8` |
| `gemma3` | `google/gemma-3-1b-it` | 1 | `--quantization fp8` |
| `qwen3` | `Qwen/Qwen3-8B` | 1 | `--quantization fp8` |
| `gemma4` | `google/gemma-4-E4B-it` | 2 | `--trust-remote-code --attention-backend TRITON_ATTN --quantization fp8` |
| `qwen25` | `Qwen/Qwen2.5-14B-Instruct` | 2 | `--quantization fp8` |
| `mistral` | `mistralai/Mistral-Small-24B-Instruct-2501` | 2 | `--quantization fp8` |
| `llama33_70b` | `meta-llama/Llama-3.3-70B-Instruct` | 4 | `--quantization fp8` |
| `qwen25_72b` | `Qwen/Qwen2.5-72B-Instruct` | 4 | `--quantization fp8` |
| `deepseekr1_70b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 4 | `--trust-remote-code --quantization fp8` |

**Card pinning** (`ZE_AFFINITY_MASK`):
- TP=1 on card N: `ZE_AFFINITY_MASK=N`
- TP=2 on cards N,N+1: `ZE_AFFINITY_MASK=N,N+1`
- TP=4 (70B models): skip `ZE_AFFINITY_MASK` and `--dtype bfloat16` entirely

**Phase grouping** (used by `--phase` flag):
- Phase 1 → TP=1 models: `llama31 deepseekr1 gemma3 qwen3`
- Phase 2 → TP=2 models: `gemma4 qwen25 mistral`
- Phase 3 → TP=4 models: `llama33_70b qwen25_72b deepseekr1_70b`

## Known Findings

- **`deepseekr1`**: FP8 KV cache causes **catastrophic accuracy collapse** (−83% on RULER at 4K, −100% on `ruler_vt`). FP8 KV is unusable for this model. Confirmed across all contexts and tasks.
- **`llama31`, `gemma3`, `qwen3`**: FP8 KV is **lossless** — all RULER task deltas ≤ 0.01 (noise).
- **`mistral` bf16 config**: runs with FP8 weight quantization (by design) to avoid OOM at 32K context on TP=2.

---

## Script 1: `run_fp8_kv_sweep.sh` — Original TP=2 Proof-of-Concept

**Purpose**: Simple sequential sweep — the original validation script before the full benchmark suite was built. Runs BF16 and FP8 KV configs per model, confirms FP8 KV works, and measures KV cache token capacity gain.

```bash
# Default: gemma4 qwen25 mistral (TP=2)
./run_fp8_kv_sweep.sh

# All 10 models sequentially
./run_fp8_kv_sweep.sh --all

# Specific models
./run_fp8_kv_sweep.sh llama31 qwen3 gemma3

# FP8 KV only for all models
./run_fp8_kv_sweep.sh --all --kv fp8
```

Output logs in `/tmp/fp8_kv_sweep/` inside the container → `~/LLM/fp8_kv_sweep/` on the host.

**Monitor**:
```bash
# Count completed runs (20 = all 10 models × 2 configs done)
ls /home/intel/LLM/fp8_kv_sweep/*.summary.txt 2>/dev/null | wc -l

# Tail the active run log
ls -t /home/intel/LLM/fp8_kv_sweep/*.log 2>/dev/null | head -1 | xargs tail -20

# Which model is currently running
docker exec vllm-test bash -c 'pgrep -a -f "vllm serve" | grep -v grep | awk "{print \$NF}"'

# See all capacity summaries so far
cat /home/intel/LLM/fp8_kv_sweep/*.summary.txt 2>/dev/null
```

---

## Script 2: `bench_fp8_kv_perf.sh` — Throughput (6 Scenarios)

**Purpose**: BF16 KV vs FP8 KV throughput across 6 real-world serving scenarios at 8K context.

```bash
# Run all 10 models sequentially
./bench_fp8_kv_perf.sh
./bench_fp8_kv_perf.sh --all    # same as above

# Run specific models only
./bench_fp8_kv_perf.sh llama31 qwen3
```

**Scenarios**: `short_decode` (128→512), `long_prefill` (4096→128), `mixed` (512→512), `high_load` (512→128), `very_long_prefill` (7168→64), `decode_heavy` (64→1024). Concurrency: 32 for most, 64 for `high_load`, 16 for `very_long_prefill`.

**Output files**: JSON results in `/tmp/fp8kv_perf/`; server/scenario/kv logs in `/tmp/fp8kv_perf/logs/`

**Monitor**:
```bash
# Count completed JSON files
ls /tmp/fp8kv_perf/*.json 2>/dev/null | wc -l

# Follow active server log
tail -f /tmp/fp8kv_perf/logs/server_llama31_bf16_tp1.log

# Check KV cache token counts
cat /tmp/fp8kv_perf/logs/kv_llama31_bf16_tp1.txt
cat /tmp/fp8kv_perf/logs/kv_llama31_fp8_tp1.txt
```

---

## Script 3: `bench_fp8_kv_sla_concurrency.sh` — SLA Max Concurrency

**Purpose**: Find the highest concurrency each model sustains while meeting: TTFT p99 ≤ 5000 ms AND TPOT p99 ≤ 200 ms.

```bash
# Full run all models
./bench_fp8_kv_sla_concurrency.sh

# Phase-filtered
./bench_fp8_kv_sla_concurrency.sh --phase 1
./bench_fp8_kv_sla_concurrency.sh --phase 2
./bench_fp8_kv_sla_concurrency.sh --phase 3
```

**Output files**:
- `sla_sweep_{TS}.csv` — every measured concurrency point (host: `/home/intel/LLM/`)
- `sla_summary_{TS}.csv` — max passing concurrency per model + FP8/BF16 ratio

**Monitor**:
```bash
watch -n 10 'wc -l /home/intel/LLM/sla_sweep_*.csv | tail -1'
tail -f /home/intel/LLM/sla_sweep_<TS>.csv
```

**Common issue**: Empty CSV (0 rows) = script crashed before first write. Check server logs:
```bash
tail -50 /home/intel/LLM/vllm_sla_server_*.log 2>/dev/null | grep -E "ERROR|OOM|killed"
```

---

## Script 4: `bench_fp8_kv_long_context.sh` — Long-Context Throughput

**Purpose**: BF16 KV vs FP8 KV throughput at 16K and 32K context. Concurrency sweep at each context length.

```bash
./bench_fp8_kv_long_context.sh --phase 1    # TP=1 models
./bench_fp8_kv_long_context.sh --phase 2    # TP=2 models
./bench_fp8_kv_long_context.sh --phase 3    # TP=4 models

# Resume a crashed model directly on specific cards/port
./bench_fp8_kv_long_context.sh --card 2 --port 8241 mistral
```

**Output files**:
- `long_context_{TS}.csv` on host
- `longctx_{model}_{config}_tp{N}_ctx{CTX}_c{CONC}_{TS}.json` in container

**Monitor**:
```bash
wc -l /home/intel/LLM/long_context_*.csv
tail -f /home/intel/LLM/long_context_<TS>.csv
```

**Key fix history**: Phase 2 had a batch-barrier bug (mistral waited for both gemma4 AND qwen25) — rewritten as slot-pipeline. The `--card`/`--port` flags were added to allow direct restart of a specific model after a crash.

---

## Script 5: `bench_fp8_kv_accuracy.sh` — RULER Accuracy Validation

**Purpose**: Does FP8 KV preserve accuracy? Runs 14 RULER tasks (NIAH variants, CWE, FWE, VT, QA) at 4K/16K/32K context per model.

```bash
./bench_fp8_kv_accuracy.sh --phase 1    # llama31 deepseekr1 gemma3 qwen3
./bench_fp8_kv_accuracy.sh --phase 2    # gemma4 qwen25 mistral
./bench_fp8_kv_accuracy.sh --phase 3    # 70B models

# Specific models only
./bench_fp8_kv_accuracy.sh llama31 qwen3

# Custom context lengths
./bench_fp8_kv_accuracy.sh --contexts 4096,16384 --phase 1
```

**Output files**:
- `ruler_summary_{TS}.csv` — live, appended after each eval (host: `/home/intel/LLM/`)
- `ruler_{model}_{config}_ctx{CTX}.log` — lm_eval output per eval
- `ruler_{model}_{config}_ctx{CTX}_{TS}/results.json` — raw lm_eval JSON in container

**Monitor**:
```bash
# Watch CSV grow (18 batches of ~13 rows for phase 2)
watch -n 30 'wc -l /home/intel/LLM/ruler_summary_*.csv | sort -k1 -n | tail -3'

# Which models currently running
docker exec vllm-test bash -c 'ps aux | grep "vllm serve" | grep -v grep | awk "{print \$NF}"'

# Tail most recent eval log
ls -t /home/intel/LLM/ruler_*.log | head -2 | xargs tail -5
```

**Known issues**:

1. **Empty CSV after run**: RULER stores `-1.0` placeholders for untested context lengths. Fixed with `val < 0` guard in extractor. If CSV is still empty, run the backfill procedure below.

2. **Intel proxy blocks HotpotQA** (`ruler_qa_hotpot`): proxy `proxy-dmz.intel.com:912` blocks the dataset download. The eval auto-retries on server restart — adds ~3 min. Not fatal.

3. **Backfill procedure** (if CSV empty after run):
```bash
docker exec vllm-test python3 -c "
import json, glob, os, re, sys

result_dir = '/tmp/fp8kv_accuracy'
TIMESTAMP = '<paste-timestamp-here>'
csv_out = f'{result_dir}/ruler_summary_{TIMESTAMP}.csv'
csv_tmp = f'/tmp/ruler_summary_{TIMESTAMP}.csv'
KNOWN_CONFIGS = {'bf16', 'fp8'}

def parse_dirname(dirname):
    if not dirname.startswith('ruler_'): return None
    rest = re.sub(r'_\d{8}_\d{6}$', '', dirname[6:])
    m = re.search(r'_ctx(\d+)$', rest)
    if not m: return None
    ctx_len = int(m.group(1)); rest = rest[:m.start()]
    parts = rest.rsplit('_', 1)
    if len(parts) != 2 or parts[1] not in KNOWN_CONFIGS: return None
    return parts[0], parts[1], ctx_len

tp_map = {'llama31':1,'deepseekr1':1,'gemma3':1,'qwen3':1,'gemma4':2,'qwen25':2,
          'mistral':2,'llama33_70b':4,'qwen25_72b':4,'deepseekr1_70b':4}
rows = []
for rfile in sorted(glob.glob(f'{result_dir}/ruler_*_{TIMESTAMP}/results.json')):
    parsed = parse_dirname(os.path.basename(os.path.dirname(rfile)))
    if not parsed: continue
    model_short, config, ctx_len = parsed
    tp = tp_map.get(model_short, 1)
    for task, metrics in json.load(open(rfile)).get('results', {}).items():
        if task == 'alias': continue
        for key, val in metrics.items():
            if not isinstance(key, str) or '_stderr' in key or key == 'alias': continue
            ctx_str = key.split(',')[0].strip()
            if not ctx_str.isdigit(): continue
            if not isinstance(val, (int, float)) or val < 0: continue
            rows.append(f'{model_short},{config},{tp},{int(ctx_str)},{task},{val:.4f}')
header = 'model,config,tp,context_length,task,score'
for path in [csv_out, csv_tmp]:
    open(path, 'w').write(header + '\n' + '\n'.join(rows) + '\n')
print(f'Written {len(rows)} rows')
"
```

4. **Validate CSV against logs** (cross-check):
```bash
# Run from host — compares every CSV row against raw eval log output
python3 /home/intel/vllm/.agents/skills/fp8-kv-bench/scripts/validate_csv.py \
    /home/intel/LLM/ruler_summary_<TS>.csv \
    /home/intel/LLM
```

---

## Stopping a Run

To cleanly stop any benchmark in progress (preserves partial CSV results):

```bash
# 1. Kill the bench orchestrator script
docker exec vllm-test bash -c 'pkill -KILL -f "bench_fp8_kv" 2>/dev/null; echo "bench script killed"'

# 2. Kill all vllm serve processes and their workers
docker exec vllm-test bash -c 'pkill -KILL -f "vllm serve" 2>/dev/null; pkill -KILL -f "from multiprocessing.spawn" 2>/dev/null; echo "vllm servers killed"'

# 3. Kill any in-flight lm_eval processes
docker exec vllm-test bash -c 'pkill -KILL -f "lm_eval" 2>/dev/null; echo "lm_eval killed"'

# 4. Kill the background watcher (if running on host)
pkill -f "watch_bench.sh" 2>/dev/null; echo "watcher killed"

# 5. Clean up card lock dirs so next run can acquire cards cleanly
docker exec vllm-test bash -c 'rm -rf /tmp/ruler_locks_* 2>/dev/null; echo "locks cleared"'
```

**Partial results are safe** — any CSV rows already written to `/home/intel/LLM/ruler_summary_<TS>.csv` are preserved. You can resume by re-running the script with the same models; it will start fresh evals (no dedup — avoid duplicate timestamps by not passing `--timestamp`).

To stop **only the watcher** without killing the benchmark:
```bash
pkill -f "watch_bench.sh"
```

---

## Background Monitoring

Every time a benchmark is launched, **also start the watcher** in a separate terminal on the host. It polls every 5 minutes, logs progress, and prints the validate + scp commands when the run completes (or diagnoses a crash).

```bash
# General form — run from HOST (not inside container)
nohup bash /home/intel/vllm/.agents/skills/fp8-kv-bench/scripts/watch_bench.sh \
    <csv_path> <expected_rows> <bench_process_pattern> [poll_sec] \
    > /home/intel/LLM/watch_<TS>.log 2>&1 &

# Example — accuracy run for qwen25 (78 rows expected)
nohup bash /home/intel/vllm/.agents/skills/fp8-kv-bench/scripts/watch_bench.sh \
    /home/intel/LLM/ruler_summary_20260522_065626.csv \
    78 bench_fp8_kv_accuracy 300 \
    > /home/intel/LLM/watch_20260522_065626.log 2>&1 &

# Tail the watcher log
tail -f /home/intel/LLM/watch_<TS>.log
```

**Expected rows by script / phase:**

| Script | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| `bench_fp8_kv_accuracy.sh` | 320 (4 models) | 78/model or 234 all-phase2 | 78/model |
| `bench_fp8_kv_sla_concurrency.sh` | varies (sweep rows) | varies | varies |
| `bench_fp8_kv_long_context.sh` | varies | varies | varies |

When the watcher exits with COMPLETE, it prints the exact `scp` and `validate` commands ready to copy-paste.

---

## Standard Workflow (Any Script)

### 0. Start the Container (first time or after host reboot)
```bash
# Run from the host inside the vllm repo directory ($(pwd) bind-mounts the scripts)
docker run --rm -td --privileged --network=host --ipc=host \
  -e http_proxy=http://proxy-dmz.intel.com:912 \
  -e https_proxy=http://proxy-dmz.intel.com:912 \
  -e no_proxy=10.0.0.0/8,habana-labs.com,.habana-labs.com,intel.com,.intel.com,127.0.0.1,localhost \
  -e HF_HOME=/tmp \
  -e HF_TOKEN=<your-hf-token> \
  -v ~/LLM:/tmp \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v $(pwd):$(pwd) \
  -w $(pwd) \
  --name=vllm-test \
  --device /dev/dri:/dev/dri \
  --entrypoint=/bin/bash \
  ghcr.io/kushal2705/vllm-xpu-v0.20-pr41689:latest
```

**Key mount points**:
- `-v ~/LLM:/tmp` → container's `/tmp` maps to `/home/intel/LLM` on the host (CSV/log files appear here live)
- `-v $(pwd):$(pwd) -w $(pwd)` → the vllm repo directory is mounted at the same path inside the container, so all scripts are available at `/home/intel/vllm/`
- `--device /dev/dri` + `-v /dev/dri/by-path` → exposes all 4 Intel Arc B70 XPU cards to the container
- `HF_TOKEN` → required for gated models (llama31, llama33_70b); replace `<your-hf-token>` with your actual token

**The container runs persistently** (`-td`) until the host reboots or you explicitly stop it with `docker stop vllm-test`.

### 1. Pre-flight
```bash
# Confirm container is running
docker ps | grep vllm-test

# Kill any leftover vllm processes from a previous run
docker exec vllm-test bash -c 'pkill -KILL -f "vllm serve" 2>/dev/null; pkill -KILL -f "from multiprocessing.spawn" 2>/dev/null; echo done'

# Enter the container
docker exec -it vllm-test bash
cd /home/intel/vllm
```

### 2. Run (always start watcher alongside)
```bash
# Step A — launch benchmark inside the container
docker exec -d vllm-test bash -c '
  cd /home/intel/vllm && \
  nohup ./bench_fp8_kv_accuracy.sh --phase 1 \
    > /tmp/bench_accuracy_phase1_$(date +%Y%m%d_%H%M%S).log 2>&1 &'

# Step B — note the CSV timestamp from the bench log, then start the watcher on HOST
nohup bash /home/intel/vllm/.agents/skills/fp8-kv-bench/scripts/watch_bench.sh \
    /home/intel/LLM/ruler_summary_<TS>.csv \
    <expected_rows> bench_fp8_kv_accuracy 300 \
    > /home/intel/LLM/watch_<TS>.log 2>&1 &
```

> The agent should extract `<TS>` from the bench log's first lines and substitute it before launching the watcher.

### 3. Monitor (from host, separate terminal)
```bash
# Live CSV row count
watch -n 15 'wc -l /home/intel/LLM/ruler_summary_*.csv 2>/dev/null | tail -3'

# Check which vllm servers are active
docker exec vllm-test bash -c 'pgrep -a -f "vllm serve" 2>/dev/null | head -5'

# Tail latest server log for OOM / crash signals
ls -t /home/intel/LLM/vllm_ruler_server_*.log | head -1 | xargs tail -20
```

### 4. Validate Results
```bash
# Accuracy benchmark: verify CSV matches log files exactly
python3 - << 'PY'
import csv, json, re, os
from collections import defaultdict
LOG_DIR  = '/home/intel/LLM'
CSV_FILE = sorted([f for f in os.listdir(LOG_DIR) if f.startswith('ruler_summary_')])[-1]
CSV_FILE = os.path.join(LOG_DIR, CSV_FILE)
# ... (see validate_csv.py script for full logic)
PY
```

### 5. Export Results to Local Machine
```bash
# From your laptop — scp everything off the B70
scp intel@b70-server-sc-3:/home/intel/LLM/ruler_summary_*.csv     ./results/
scp intel@b70-server-sc-3:/home/intel/LLM/long_context_*.csv      ./results/
scp intel@b70-server-sc-3:/home/intel/LLM/sla_summary_*.csv       ./results/
scp intel@b70-server-sc-3:/home/intel/LLM/sla_sweep_*.csv         ./results/
```

---

## Progress Tracking

The skill itself does not track run state — but the agent can check progress at any time by inspecting live artifacts:

| What to check | Command |
|---|---|
| How many accuracy evals done | `wc -l /home/intel/LLM/ruler_summary_*.csv` (320 rows = phase 1 complete) |
| Which models finished accuracy | `awk -F, 'NR>1 {print $1","$2}' /home/intel/LLM/ruler_summary_*.csv \| sort -u` |
| Long-context progress | `wc -l /home/intel/LLM/long_context_*.csv` |
| SLA sweep progress | `wc -l /home/intel/LLM/sla_sweep_*.csv \| tail -1` |
| Active vllm servers | `docker exec vllm-test pgrep -c -f "vllm serve" 2>/dev/null` |
| Scheduler still running | `docker exec vllm-test pgrep -f "bench_fp8_kv" 2>/dev/null` |

For persistent cross-session progress tracking, ask the agent to write a status note to `/memories/session/fp8-bench-status.md`.

---

## Adding a New Model

1. Add shortname → HF model ID to `resolve_model()` in the relevant script(s)
2. Add TP to `resolve_tp()` (1, 2, or 4)
3. Add any required extra flags to `resolve_extra_args()` (e.g., `--trust-remote-code`)
4. Add the model to `ALL_MODELS` array
5. Add TP entry to `tp_map` in the backfill script and `validate_csv.py`
6. Update this skill's Model Registry table

---

## Completed Runs (as of May 2026)

| Script | Phase 1 (TP=1) | Phase 2 (TP=2) | Phase 3 (TP=4) |
|---|---|---|---|
| `bench_fp8_kv_perf.sh` | ⏳ | ⏳ | ✅ deepseekr1_70b |
| `bench_fp8_kv_sla_concurrency.sh` | ✅ | ⏳ | ✅ deepseekr1_70b |
| `bench_fp8_kv_long_context.sh` | ✅ | ⏳ partial | ✅ deepseekr1_70b |
| `bench_fp8_kv_accuracy.sh` | ✅ validated | ⏳ ready | ⏳ |
| `run_fp8_kv_sweep.sh` | N/A | ✅ | N/A |
