# FP8 KV Cache Benchmark Guide

Benchmarks that validate FP8 KV cache quantization vs BF16 baseline on Intel
B-series GPUs (B70) under vLLM XPU.

| Script | What it measures |
|---|---|
| `run_fp8_kv_sweep.sh` | Smoke test + capacity summary: BF16 vs FP8 KV, TP=2, 3 large models |
| `bench_fp8_kv_perf.sh` | Full perf sweep: BF16 vs FP8 KV across 6 serving scenarios, 10 models, TP=1/2/4 |
| `bench_fp8_kv_sla_concurrency.sh` | SLA-bound max concurrency: finds highest concurrency where TTFT(p99)≤5s and TPOT(p99)≤200ms |
| `bench_fp8_kv_long_context.sh` | Long-context capacity: max concurrency at 16K and 32K context, BF16 vs FP8 KV, 10 models |
| `bench_fp8_kv_accuracy.sh` | Accuracy validation: RULER suite (NIAH + CWE + FWE + VT + QA) at 4K/16K/32K, BF16 vs FP8 KV, 10 models |

---

## Background: What is FP8 KV Cache?

During inference, the key-value (KV) cache stores intermediate attention tensors
for every token in the context. By default these are stored in BF16 (2 bytes per
element). With `--kv-cache-dtype fp8`, vLLM stores them in FP8 (1 byte per
element), giving:

- **~2× KV cache capacity** — same GPU memory holds twice as many tokens
- **~2× maximum concurrency** — more requests fit simultaneously
- **Minimal accuracy loss** — FP8 E4M3 covers the dynamic range of KV activations well

The flag applies only to the KV cache; model weights remain in BF16 (or FP8 if
`--quantization fp8` is also set).

---

## Prerequisites

- Intel GPU with XPU drivers installed (B70: 4× cards, 32 GiB GDDR6 each = 128 GiB total)
- vLLM XPU build installed in the environment
- `curl`, `python3` available in the environment
- HuggingFace model weights cached at `$HF_HOME` (default `/tmp`)

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
  ghcr.io/kushal2705/vllm-xpu-v0.20-pr41689:latest
```

Key bind mounts:
- `-v ~/LLM:/tmp` — models at `~/LLM/hub/` appear as `/tmp/hub/` in container; each script writes results to its own `/tmp/fp8kv_*/` directory, which appears as `~/LLM/fp8kv_*/` on the host
- `-v /home/intel/LLM/xpu_cache:/root/.cache` — persists the XPU/IGC kernel compilation cache across container restarts (~125 GB on first run; reused on subsequent runs)
- `-v $(pwd):$(pwd) -w $(pwd)` — makes the script directory available at the same path inside the container

### Environment variables set by the scripts

| Variable | Value | Purpose |
|---|---|---|
| `VLLM_TARGET_DEVICE` | `xpu` | Target Intel XPU backend |
| `VLLM_MLA_DISABLE` | `1` | Disable Multi-head Latent Attention (not supported on XPU) |
| `VLLM_USE_V1` | `1` | Use vLLM V1 engine |
| `VLLM_ENGINE_READY_TIMEOUT` | `900` | Seconds to wait for engine init |
| `VLLM_NO_USAGE_STATS` | `1` | Disable telemetry |
| `ZE_AFFINITY_MASK` | `0`, `0,1`, `2,3`, etc. | Pin process to specific XPU card(s) |

---

## Script 1 — `run_fp8_kv_sweep.sh`

### Purpose

A **validation sweep**: for each model, starts a BF16 server then an FP8 server,
extracts capacity metrics from the server log, runs a smoke test (single chat
request), and stops the server. No throughput benchmarking — the goal is to
confirm FP8 KV works and measure the KV cache token capacity gain.

### Models

| Model | Size | Notes |
|---|---|---|
| `google/gemma-4-E4B-it` | E4B MoE | Requires `--trust-remote-code`, forces TRITON_ATTN |
| `Qwen/Qwen2.5-14B` | 14B | — |
| `mistralai/Mistral-Small-24B-Instruct-2501` | 24B | — |

### Server arguments

```
--dtype bfloat16
--tensor-parallel-size 2
--max-model-len 4096
--max-num-batched-tokens 8192
--max-num-seqs 128
--gpu-memory-utilization 0.9
--quantization fp8          # model weights in FP8
--enforce-eager
--trust-remote-code
--no-enable-prefix-caching
--block-size 64
--port 8000
--kv-cache-dtype fp8        # added only for the fp8 KV run
```

### Per-run flow

```
start server (setsid, new process group)
  └── wait for /health (up to 900s, polling every 5s)
  └── grep log → extract capacity summary
  └── smoke test: POST /v1/chat/completions  "Hello" → max_tokens=8
  └── stop server (kill by process group)
```

### Output files

| File | Contents |
|---|---|
| `./logs/fp8_kv_sweep_tp2/<model>__kv_<dtype>_tp2.log` | Full server stdout/stderr |
| `./logs/fp8_kv_sweep_tp2/<model>__kv_<dtype>_tp2.summary.txt` | Extracted capacity lines |

### Capacity lines extracted

```
Model loading took X GiB memory and Y seconds
Available KV cache memory: X GiB
GPU KV cache size: X tokens
Maximum concurrency for 4,096 tokens per request: Xx
Checkpoint size: X GiB
```

### How to run

```bash
# Run from inside the vLLM container
chmod +x run_fp8_kv_sweep.sh
./run_fp8_kv_sweep.sh
```

Runs all 3 models sequentially (bf16 then fp8 for each). Logs go to
`./logs/fp8_kv_sweep_tp2/`.

---

## Script 2 — `bench_fp8_kv_perf.sh`

### Purpose

A **full performance benchmark**: for each model and each KV dtype (bf16, fp8),
starts a server, warms it up, runs 6 serving scenarios using `vllm bench serve`,
saves JSON results, and prints a comparison table. Also benchmarks TP=1 vs TP=2
for large models.

### Models and TP assignment

| Shorthand | Model | TP configs | Extra flags |
|---|---|---|---|
| `llama31` | `meta-llama/Llama-3.1-8B-Instruct` | TP=1 | `--quantization fp8` |
| `deepseekr1` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | TP=1 | `--trust-remote-code --quantization fp8` |
| `gemma3` | `google/gemma-3-1b-it` | TP=1 | `--quantization fp8` |
| `qwen3` | `Qwen/Qwen3-8B` | TP=1 | `--quantization fp8` |
| `gemma4` | `google/gemma-4-E4B-it` | TP=1, TP=2 | `--trust-remote-code --attention-backend TRITON_ATTN --quantization fp8` |
| `qwen25` | `Qwen/Qwen2.5-14B-Instruct` | TP=1, TP=2 | `--quantization fp8` |
| `mistral` | `mistralai/Mistral-Small-24B-Instruct-2501` | TP=2 only | `--quantization fp8` |
| `llama33_70b` | `meta-llama/Llama-3.3-70B-Instruct` | TP=4 only | `--quantization fp8` |
| `qwen25_72b` | `Qwen/Qwen2.5-72B-Instruct` | TP=4 only | `--quantization fp8` |
| `deepseekr1_70b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | TP=4 only | `--trust-remote-code --quantization fp8` |
| `llama31_fp8` | `nvidia/Llama-3.1-8B-Instruct-FP8` | TP=1 only | _(weights pre-quantized FP8; runs BF16 KV only — FP8 KV skipped)_ |

All models use `--quantization fp8` (FP8 weight quantization) except `llama31_fp8` (weights already statically FP8 — no flag needed). Small models (≤8B)
fit on one B70 card; mid-size models (14–24B) run TP=1 and/or TP=2; 70B/72B models
require 4 cards and run TP=4 only.

### KV dtype configs

| Config | `--kv-cache-dtype` flag | Description |
|---|---|---|
| `bf16` | _(omitted — default)_ | 16-bit KV cache baseline |
| `fp8` | `--kv-cache-dtype fp8` | 8-bit KV cache; ~2× capacity |

### 6 Serving scenarios

| Scenario | ISL | OSL | Prompts | Concurrency | Stress profile |
|---|---|---|---|---|---|
| `short_decode` | 128 | 512 | 200 | 32 | Typical chat |
| `long_prefill` | 4096 | 128 | 200 | 32 | Document ingestion |
| `mixed` | 512 | 512 | 200 | 32 | Balanced |
| `high_load` | 512 | 128 | 500 | 64 | High concurrency |
| `very_long_prefill` | 7168 | 64 | 200 | 16 | Max-context prefill |
| `decode_heavy` | 64 | 1024 | 200 | 32 | Long generation |

`max_model_len=8192`. Any scenario where ISL+OSL > 8192 is automatically skipped.

### Server arguments

```
--dtype bfloat16            # omitted for llama33_70b and qwen25_72b (use auto)
--tensor-parallel-size <1, 2, or 4>
--max-model-len 8192
--gpu-memory-utilization 0.92
--enforce-eager
--max-num-batched-tokens 8192
--max-num-seq 64
--block-size 64
--no-enable-log-requests
--no-enable-prefix-caching
--quantization fp8          # model weights in FP8 (applied to all models)
--kv-cache-dtype fp8        # fp8 runs only
```

Special per-model flags:
- `gemma4`: `--trust-remote-code --attention-backend TRITON_ATTN`
- `deepseekr1`, `deepseekr1_70b`: `--trust-remote-code`
- All models: `--quantization fp8` (FP8 weight quantization applied universally)

### Per-run flow

```
for each TP in tp_list:
  for each config in [bf16, fp8]:
    start server (setsid, new process group, ZE_AFFINITY_MASK pinned*)
      └── wait for /health (up to 900s, polling every 1s)
      └── warmup: 3 requests, concurrency 4
    extract KV cache token count from server log
    for each of 6 scenarios:
      vllm bench serve → save JSON to RESULT_DIR
    stop server (kill by process group)
print summary tables
```

### Card allocation

The workstation has **4 cards (cards 0–3)**. Each model run is allocated all 4
cards, so only **one model can run at a time**. TP determines how many of the 4
cards are actually used:

| TP | ZE_AFFINITY_MASK | Cards used |
|---|---|---|
| TP=1 | `0` | 1 of 4 |
| TP=2 | `0,1` | 2 of 4 |
| TP=4 | `0,1,2,3` | all 4 |

\* `llama33_70b`, `qwen25_72b`, and `deepseekr1_70b` skip `ZE_AFFINITY_MASK` and let vLLM auto-detect all available cards.

For this reason, **parallel multi-model mode is not suitable** on this workstation.
Use `--all` (sequential) or run models one at a time.

### Output files

| File | Contents |
|---|---|
| `/tmp/fp8kv_perf/<model>_<config>_tp<N>__<scenario>__<ts>.json` | Throughput + latency metrics |
| `/tmp/fp8kv_perf/logs/scenario_<model>_<config>_tp<N>__<scenario>__<ts>.log` | `vllm bench serve` stdout/stderr |
| `/tmp/fp8kv_perf/logs/server_<model>_<config>_tp<N>.log` | vLLM server stdout/stderr (separate file per config) |
| `/tmp/fp8kv_perf/logs/kv_<model>_<config>_tp<N>.txt` | GPU KV cache token count |

### Metrics in each JSON result

| Metric | Description |
|---|---|
| `request_throughput` | Requests/second |
| `output_throughput` | Output tokens/second |
| `mean_ttft_ms` | Mean Time To First Token (ms) |
| `mean_tpot_ms` | Mean Time Per Output Token (ms) |
| `mean_itl_ms` | Mean Inter-Token Latency (ms) |
| `p90_ttft_ms` / `p99_ttft_ms` | p90/p99 TTFT percentiles |
| `p90_tpot_ms` | p90 TPOT percentile |

### How to run

```bash
chmod +x bench_fp8_kv_perf.sh

# Run all 11 models (including llama31_fp8); TP=1 in parallel, TP=2 in pairs, TP=4 sequential
./bench_fp8_kv_perf.sh
./bench_fp8_kv_perf.sh --all    # same as above

# Run specific models only
./bench_fp8_kv_perf.sh llama31 qwen3 gemma3
```

### Watching progress

```bash
# Follow the server log for a model (separate file per config+tp)
tail -f /tmp/fp8kv_perf/logs/server_llama31_bf16_tp1.log
tail -f /tmp/fp8kv_perf/logs/server_llama31_fp8_tp1.log

# Follow a specific bench run log
tail -f /tmp/fp8kv_perf/logs/scenario_llama31_bf16_tp1__short_decode__*.log

# Check KV cache token counts after runs complete
cat /tmp/fp8kv_perf/logs/kv_llama31_bf16_tp1.txt
cat /tmp/fp8kv_perf/logs/kv_llama31_fp8_tp1.txt
```

### Summary output (printed to stdout)

After all runs for a model complete, the script prints:

```
================================================================
  SUMMARY: meta-llama/Llama-3.1-8B-Instruct (llama31)
================================================================

  ── TP=1 ──────────────────────────────────────────────────────

--- KV Cache Tokens (TP=1) ---
  Config     KV Cache Toks      vs bf16
  --------   --------------   ------------
  bf16            1234567            -
  fp8             2469134        2.00x

--- short_decode  (ISL=128, OSL=512, N=200, C=32, TP=1) ---
  Config      Req/s   OutTok/s       TTFT       TPOT        ITL   p90 TTFT   p90 TPOT   p99 TTFT
  --------   ------  ----------  ---------  ---------  --------  ---------  ---------  ---------
  bf16         12.3        6300        145       52.3      52.1        210       58.1        380
  fp8          13.1        6700        138       49.8      49.5        198       55.2        361
  ...
```

---

## Script 3 — `bench_fp8_kv_sla_concurrency.sh`

### Purpose

An **SLA-bound max concurrency sweep**: for each model and KV dtype (bf16, fp8),
finds the highest concurrency level where **both** latency SLAs hold simultaneously:

- `TTFT(p99) ≤ 5000 ms`
- `TPOT(p99) ≤ 200 ms`

Reports the max passing concurrency and the **ratio vs BF16** — showing how many
more simultaneous users FP8 KV cache can serve within the same SLA envelope.

### Models and TP assignment

Same as Script 2 (`gemma4` and `qwen25` run both TP=1 and TP=2; `mistral` TP=2 only; 70B models TP=4 only).

All models receive `--quantization fp8` (FP8 weight quantization) via the wildcard in `resolve_extra_args()`. The extra flags column below shows only **model-specific additions** beyond the universal `--quantization fp8`:

| Shorthand | Model | TP configs | Additional flags |
|---|---|---|---|
| `llama31` | `meta-llama/Llama-3.1-8B-Instruct` | TP=1 | — |
| `deepseekr1` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | TP=1 | `--trust-remote-code` |
| `gemma3` | `google/gemma-3-1b-it` | TP=1 | — |
| `qwen3` | `Qwen/Qwen3-8B` | TP=1 | — |
| `gemma4` | `google/gemma-4-E4B-it` | TP=1, TP=2 | `--trust-remote-code --attention-backend TRITON_ATTN` |
| `qwen25` | `Qwen/Qwen2.5-14B-Instruct` | TP=1, TP=2 | — |
| `mistral` | `mistralai/Mistral-Small-24B-Instruct-2501` | TP=2 only | — |
| `llama33_70b` | `meta-llama/Llama-3.3-70B-Instruct` | TP=4 only | — |
| `qwen25_72b` | `Qwen/Qwen2.5-72B-Instruct` | TP=4 only | — |
| `deepseekr1_70b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | TP=4 only | `--trust-remote-code` |

### KV dtype configs

| Config | Flag | Description |
|---|---|---|
| `bf16` | _(omitted)_ | BF16 KV cache baseline |
| `fp8` | `--kv-cache-dtype fp8` | FP8 KV cache; ~2× capacity |

### Sweep parameters

| Parameter | Value |
|---|---|
| Concurrency levels | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| Early exit | After 2 consecutive SLA failures |
| ISL / OSL | 1024 / 512 tokens |
| `max_model_len` | 4096 |
| SLA TTFT(p99) | ≤ 5000 ms |
| SLA TPOT(p99) | ≤ 200 ms |

### Per-run flow

```
for each TP in tp_list:
  for each config in [bf16, fp8]:
    start server (setsid, new process group)
      └── wait for /health (up to 900s)
      └── warmup: 4 requests, concurrency 4
    for each concurrency in [1,2,4,8,16,32,64,128,256]:
      vllm bench serve → extract p99_ttft, p99_tpot, output_tps
      if pass: record max_pass; reset fail counter
      if fail: increment fail counter; break if ≥ 2
    record max_pass → /tmp/sla_max_<model>_<config>_tp<N>.txt
    stop server
emit CSV summary (max passing conc + ratio vs bf16)
```

### Scheduling

- **Phase 1**: TP=1-only models (`llama31`, `deepseekr1`, `gemma3`, `qwen3`) run **in parallel**, each pinned to a separate card (up to 4 simultaneous).
- **Phase 2**: TP=2 models (`gemma4`, `qwen25`, `mistral`) run **in parallel pairs** — slot 0 on cards 0,1 and slot 1 on cards 2,3. Two models run simultaneously; waits for both before launching the next pair.
- **Phase 3**: TP=4 models (`llama33_70b`, `qwen25_72b`, `deepseekr1_70b`) run **sequentially** — each needs all 4 cards.

### Output files

| File | Contents |
|---|---|
| `/tmp/fp8kv_sla/sla_sweep_<TS>.csv` (host: `~/LLM/fp8kv_sla/`) | Every measured point: model, config, tp, concurrency, p99_ttft, p99_tpot, output_tps, sla_pass |
| `/tmp/fp8kv_sla/sla_summary_<TS>.csv` (host: `~/LLM/fp8kv_sla/`) | Max passing concurrency + ratio vs bf16 per (model, config, tp) |
| `/tmp/fp8kv_sla/logs/vllm_sla_server_<model>_<config>_tp<N>.log` | vLLM server stdout/stderr |
| `/tmp/fp8kv_sla/logs/bench_sla_<model>_<config>_tp<N>_c<C>_<TS>.log` | `vllm bench serve` stdout per concurrency point |

> **Note**: During the run, per-point rows are appended live to `${LOG_DIR}/sla_sweep_<TS>.csv` (`~/LLM/fp8kv_sla/logs/`). At the end, `emit_summary()` consolidates them into the final `~/LLM/fp8kv_sla/sla_sweep_<TS>.csv`.

### How to run

```bash
chmod +x bench_fp8_kv_sla_concurrency.sh

# Run all 10 models (TP=1 parallel ×4, TP=2 in pairs, TP=4 sequential)
./bench_fp8_kv_sla_concurrency.sh

# TP=1 models only (pass model names directly — no --phase flag)
./bench_fp8_kv_sla_concurrency.sh llama31 deepseekr1 gemma3 qwen3

# TP=2 models only
./bench_fp8_kv_sla_concurrency.sh gemma4 qwen25 mistral

# Override KV configs
./bench_fp8_kv_sla_concurrency.sh --configs fp8 llama31 qwen3
```

### Example summary output

```
================================================================
  SLA: TTFT(p99) ≤ 5000ms  AND  TPOT(p99) ≤ 200ms
================================================================

model,config,tp,max_conc_passing_sla,ratio_vs_bf16
llama31,bf16,1,64,-
llama31,fp8,1,128,2.00x
qwen25,bf16,2,32,-
qwen25,fp8,2,64,2.00x
...
```

A ratio of **2.0×** means FP8 KV cache doubles the number of users that can be
served concurrently within the SLA — consistent with the 2× KV token capacity gain.

---

## Script 4 — `bench_fp8_kv_long_context.sh`

### Purpose

A **long-context capacity benchmark**: for each model, TP config, and KV dtype
(bf16, fp8), starts a server at 16K or 32K `max_model_len` and sweeps concurrency
to find the highest level achievable without server crash or bench failure.

BF16 is included as a **canary**: at 16K–32K context the KV cache alone can
exhaust available VRAM, causing OOM at server init. Those cases are recorded as
`OOM` in the CSV, making the FP8 advantage explicit.

**OOM auto-retry**: if a server fails to start, the script automatically retries
with `--quantization fp8` added (FP8 model weights, ~2× VRAM reduction). If it
still fails, the configuration is recorded as `OOM` and the script moves on.

### Models and TP assignment

| Shorthand | Model | TP configs | Extra flags |
|---|---|---|---|
| `llama31` | `meta-llama/Llama-3.1-8B-Instruct` | TP=1 | — |
| `deepseekr1` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | TP=1 | `--trust-remote-code` |
| `gemma3` | `google/gemma-3-1b-it` | TP=1 | — |
| `qwen3` | `Qwen/Qwen3-8B` | TP=1 | — |
| `gemma4` | `google/gemma-4-E4B-it` | TP=1, TP=2 | `--trust-remote-code --attention-backend TRITON_ATTN` |
| `qwen25` | `Qwen/Qwen2.5-14B-Instruct` | TP=1, TP=2 | — |
| `mistral` | `mistralai/Mistral-Small-24B-Instruct-2501` | TP=2 only | `--quantization fp8` |
| `llama33_70b` | `meta-llama/Llama-3.3-70B-Instruct` | TP=4 only | `--quantization fp8` |
| `qwen25_72b` | `Qwen/Qwen2.5-72B-Instruct` | TP=4 only | `--quantization fp8` |
| `deepseekr1_70b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | TP=4 only | `--trust-remote-code --quantization fp8` |

### KV dtype configs

| Config | Flag | Description |
|---|---|---|
| `bf16` | _(omitted)_ | BF16 KV cache baseline |
| `fp8` | `--kv-cache-dtype fp8` | FP8 KV cache; ~2× capacity |

### Benchmark parameters

| Parameter | Value |
|---|---|
| Context lengths | 16384, 32768 |
| Concurrency sweep | 1, 2, 4, 8, 16, 32, 64 |
| ISL | `ctx_len / 4` (4096 at 16K, 8192 at 32K) |
| OSL | **512** (capped — prefill capacity focus; uncapped ctx/8 would add 20–40 min per point) |
| `max_model_len` | `ctx_len` |
| `max_num_batched_tokens` | `max(ctx_len, 8192)` |
| `max_num_seq` | 32 |

### Per-run flow

```
for each TP in tp_list:
  for each ctx_len in [16384, 32768]:
    for each config in [bf16, fp8]:
      start server (setsid, new process group)
        └── if OOM: retry with --quantization fp8
        └── if still OOM: record OOM → skip
      warmup: 2 requests, concurrency 1
      for each concurrency in [1,2,4,8,16,32,64]:
        vllm bench serve (n_prompts = min(max(conc×2, 16), 256))
          → save JSON to RESULT_DIR
        if server died (rc=2): stop sweep
        if bench failed (rc=1): stop sweep
      record max successful concurrency
      stop server
emit CSV
```

### Scheduling

- **Phase 1**: TP=1-only models (`llama31`, `deepseekr1`, `gemma3`, `qwen3`) run **in parallel**, each pinned to a separate card.
- **Phase 2**: TP=2-capable models (`gemma4`, `qwen25`, `mistral`) run **in parallel pairs** — slot 0 on cards 0,1 (port BASE_PORT) and slot 1 on cards 2,3 (port BASE_PORT+1). Batches of 2 are launched at a time; waits for both before starting the next pair.
- **Phase 3**: TP=4 models (`llama33_70b`, `qwen25_72b`, `deepseekr1_70b`) run **sequentially** — each needs all 4 cards.

### Output files

| File | Contents |
|---|---|
| `/tmp/fp8kv_long_context/long_context_<TS>.csv` (host: `~/LLM/fp8kv_long_context/`) | All measured points (live-appended) |
| `~/LLM/long_context_<TS>.csv` | Live host copy appended in real time (same data) |
| `/tmp/fp8kv_long_context/logs/vllm_longctx_server_<model>_<config>_tp<N>_ctx<L>.log` | vLLM server stdout/stderr |
| `/tmp/fp8kv_long_context/logs/bench_longctx_<model>_<config>_tp<N>_ctx<L>_c<C>_<TS>.log` | `vllm bench serve` stdout per point |

### CSV columns

| Column | Description |
|---|---|
| `model` | Model shorthand |
| `config` | `bf16` or `fp8` |
| `tp` | Tensor parallel size |
| `context_length` | 16384 or 32768 |
| `concurrency` | Concurrency level tested |
| `num_prompts` | Number of requests sent |
| `request_throughput` | Requests/second |
| `output_throughput` | Output tokens/second |
| `mean_ttft_ms` / `p99_ttft_ms` | Time-to-first-token (mean, p99) |
| `mean_tpot_ms` / `p99_tpot_ms` | Time-per-output-token (mean, p99) |
| `status` | `OK`, `FAIL`, or `OOM` |

### How to run

```bash
chmod +x bench_fp8_kv_long_context.sh

# Run all 10 models (default)
./bench_fp8_kv_long_context.sh

# Single model
./bench_fp8_kv_long_context.sh llama31

# Specific models
./bench_fp8_kv_long_context.sh gemma4 qwen25 mistral

# FP8 KV only (skip BF16 baseline)
./bench_fp8_kv_long_context.sh --configs fp8 llama31
```

### Expected outcomes

- **Small models (≤8B, TP=1)**: BF16 and FP8 KV both start. FP8 KV reaches higher concurrency before degrading.
- **Mid-size models (14–24B) at 32K, TP=1**: BF16 likely OOMs at init; auto-retry with `--quantization fp8` may recover. FP8 KV (weights BF16) should start with a smaller KV footprint.
- **70B models (TP=4)**: weights already use `--quantization fp8`. FP8 KV further reduces the KV footprint, enabling more concurrent long-context requests.

---

## Script 5 — `bench_fp8_kv_accuracy.sh`

### Purpose

An **accuracy validation benchmark**: for each model and KV dtype (bf16, fp8),
starts a vLLM server and runs the full **RULER** evaluation suite from
`lm-eval-harness` at 4K, 16K, and 32K context lengths.

RULER measures long-context comprehension across several task types:
- **NIAH** (Needle-in-a-Haystack) variants: single/multi-key/value/query retrieval
- **CWE** (Common-Word Extraction)
- **FWE** (Frequent-Word Extraction)
- **VT** (Variable Tracking)
- **QA** (Question Answering)

The goal is to confirm that FP8 KV cache does **not degrade accuracy** vs BF16 —
i.e. the score delta should be near zero for all tasks and context lengths.

BF16 is included as a **canary**: at 16K–32K context on a single card, BF16 may
OOM at server init, making the FP8 KV cache advantage explicit in the CSV.

**OOM auto-retry**: same as `bench_fp8_kv_long_context.sh` — if a server fails,
the script retries with `--quantization fp8` (model weights in FP8, ~2× VRAM
reduction). If still OOM, the run is recorded as `OOM`.

### Models and TP assignment

Each model uses a single **canonical TP** for accuracy evaluation. Accuracy is
independent of TP; the canonical TP is chosen to avoid OOM at long contexts.

| Shorthand | Model | Canonical TP | Extra flags |
|---|---|---|---|
| `llama31` | `meta-llama/Llama-3.1-8B-Instruct` | TP=1 | — |
| `deepseekr1` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | TP=1 | `--trust-remote-code` |
| `gemma3` | `google/gemma-3-1b-it` | TP=1 | — |
| `qwen3` | `Qwen/Qwen3-8B` | TP=1 | — |
| `gemma4` | `google/gemma-4-E4B-it` | TP=2 | `--trust-remote-code --attention-backend TRITON_ATTN` |
| `qwen25` | `Qwen/Qwen2.5-14B-Instruct` | TP=2 | — |
| `mistral` | `mistralai/Mistral-Small-24B-Instruct-2501` | TP=2 | `--quantization fp8` |
| `llama33_70b` | `meta-llama/Llama-3.3-70B-Instruct` | TP=4 | `--quantization fp8` |
| `qwen25_72b` | `Qwen/Qwen2.5-72B-Instruct` | TP=4 | `--quantization fp8` |
| `deepseekr1_70b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | TP=4 | `--trust-remote-code --quantization fp8` |

### KV dtype configs

| Config | Flag | Description |
|---|---|---|
| `bf16` | _(omitted — default)_ | 16-bit KV cache baseline |
| `fp8` | `--kv-cache-dtype fp8` | 8-bit KV cache; ~2× capacity |

### Benchmark parameters

| Parameter | Value |
|---|---|
| Context lengths | 4096, 16384, 32768 |
| RULER tasks | `ruler` (full suite) |
| Sample limit | 50 (TP=1 models), 25 (TP=2/4 models) |
| `max_model_len` | `ctx_len` |
| `max_num_seqs` | 2 (lm_eval sends requests sequentially) |
| `max_num_batched_tokens` | `max(ctx_len, 8192)` |
| `num_concurrent` (lm_eval) | 1 |
| Request timeout | 1800s |

### Per-run flow

```
scheduler dispatches jobs as cards become free:
  for each model (assigned card/port by scheduler):
    for each ctx_len in [4096, 16384, 32768]:
    for each config in [bf16, fp8]:
      start server (setsid, new process group, ZE_AFFINITY_MASK pinned*)
        └── if OOM: retry with --quantization fp8
        └── if still OOM: record OOM → skip
      run lm_eval RULER suite → save results.json to RESULT_DIR
        └── on failure: restart server and retry once
        └── if still fails: record UNSTABLE → skip
      extract per-task scores → append to raw CSV
      stop server
emit summary CSV + comparison table
```

\* 70B models skip `ZE_AFFINITY_MASK` and let vLLM auto-detect all available cards.

### Scheduling

A **unified resource-aware scheduler** manages all 10 models in a single queue.
Every 5 seconds it scans pending jobs and greedily dispatches each one to the
first available card group that satisfies its TP requirement:

| TP | Cards needed | Port |
|---|---|---|
| TP=1 | any one free card (0, 1, 2, or 3) | `BASE_PORT + card` |
| TP=2 | first free pair: cards 0,1 **or** cards 2,3 | `BASE_PORT` or `BASE_PORT+1` |
| TP=4 | all 4 cards free simultaneously | `BASE_PORT` |

Card locks use atomic `mkdir` so no two jobs ever share a card. A job is
dispatched **the moment** the required cards become free — a fast TP=1 model
immediately unlocks its card for the next job; a TP=2 job starts as soon as
any free pair appears; TP=4 waits until the full board is clear.

**Example timeline with all 10 models:**
```
t=0:   llama31(TP=1)→card0   deepseekr1(TP=1)→card1
       gemma3(TP=1)→card2    qwen3(TP=1)→card3
       [gemma4, qwen25, mistral, 70B models waiting]

t=+Δ:  gemma3 done → card2,3 both free → qwen25(TP=2) dispatched on cards 2,3
t=+Δ:  llama31 done → card0,1 both free → gemma4(TP=2) dispatched on cards 0,1
t=+Δ:  deepseekr1/qwen3 done → mistral(TP=2) picks up next free pair
t=+Δ:  all TP=2 done → llama33_70b(TP=4) dispatched on all 4 cards
       → qwen25_72b(TP=4) → deepseekr1_70b(TP=4) sequentially
```

The `--phase` flag **filters** the input model list before the scheduler runs:
- `--phase 1`: only TP=1 models → up to 4 run in parallel
- `--phase 2`: only TP=2 models → up to 2 run in parallel
- `--phase 3`: only TP=4 models → sequential (needs all cards)

### Output files

| File | Contents |
|---|---|
| `/tmp/fp8kv_accuracy/ruler_summary_<TS>.csv` | All per-task scores: model, config, tp, context_length, task, score |
| `/tmp/ruler_summary_<TS>.csv` | Live copy appended after each eval (same data, updated as evals complete) |
| `/tmp/fp8kv_accuracy/ruler_<model>_<config>_ctx<L>_<TS>/results.json` | Full lm_eval output per run |
| `/tmp/ruler_<model>_<config>_ctx<L>.log` | lm_eval stdout/stderr per run |
| `/tmp/vllm_ruler_server_<model>_<config>_tp<N>_ctx<L>.log` | vLLM server stdout/stderr |

### CSV columns

| Column | Description |
|---|---|
| `model` | Model shorthand |
| `config` | `bf16` or `fp8` |
| `tp` | Tensor parallel size |
| `context_length` | 4096, 16384, or 32768 |
| `task` | RULER task name (e.g. `ruler_niah_single_1`) |
| `score` | Task accuracy (0.0–1.0), or `OOM` / `UNSTABLE` |

### How to run

```bash
chmod +x bench_fp8_kv_accuracy.sh

# All 10 models — scheduler fills cards dynamically
./bench_fp8_kv_accuracy.sh

# Phase 1: TP=1 models only (llama31, deepseekr1, gemma3, qwen3) — 4 parallel
./bench_fp8_kv_accuracy.sh --phase 1

# Phase 2: TP=2 models only (gemma4, qwen25, mistral) — 2 parallel
./bench_fp8_kv_accuracy.sh --phase 2

# Phase 3: TP=4 models only (llama33_70b, qwen25_72b, deepseekr1_70b) — sequential
./bench_fp8_kv_accuracy.sh --phase 3

# Single model, fp8 KV only (skip BF16 canary)
./bench_fp8_kv_accuracy.sh --configs fp8 llama31

# Specific models (scheduler still applies)
./bench_fp8_kv_accuracy.sh gemma4 mistral

# Short context only (faster validation)
./bench_fp8_kv_accuracy.sh --contexts 4096,16384 llama31
```

### Example summary output

```
================================================================
  RULER Accuracy: BF16 KV vs FP8 KV Cache
================================================================

Model          TP    Ctx  Task                    BF16   FP8-KV     Delta
------------------------------------------------------------------------------
llama31         1   4096  ruler_niah_single_1     0.9800   0.9800   +0.0000
llama31         1  16384  ruler_niah_single_1     0.9600   0.9560   -0.0040
llama31         1  32768  ruler_niah_single_1     0.8800   0.8760   -0.0040
...
qwen25          2  16384  ruler_niah_single_1     0.9200   0.9160   -0.0040
qwen25          2  32768  ruler_niah_single_1      OOM    0.8920  BF16 OOM
...
```

A delta near **0.00** confirms FP8 KV cache preserves accuracy. Rows where
`BF16 = OOM` and `FP8-KV` has a valid score show where FP8 KV enables
long-context inference that BF16 cannot even start.

### Expected outcomes

- **All models at 4K**: BF16 and FP8 KV both run; scores should be identical or within noise (|delta| < 0.01).
- **Small models (≤8B) at 16K–32K**: Both configs start; FP8 KV score matches BF16 within noise.
- **Mid-size models (14–24B, TP=2) at 32K**: BF16 may start (fits with TP=2); FP8 KV should also start with a smaller KV footprint.
- **Mid-size models at 32K, TP=1**: BF16 will OOM; auto-retry adds `--quantization fp8` to weights. FP8 KV with FP8 weights may start and produce a valid score (BF16 baseline will be `OOM`).
- **70B models (TP=4) at all contexts**: weights already use `--quantization fp8`. FP8 KV further reduces KV footprint, enabling longer-context eval that BF16 KV may fail.

---

## Expected FP8 KV capacity gain

On a B70 (32 GiB GDDR6), with `gpu-memory-utilization=0.92`:

| Model | TP | BF16 KV tokens | FP8 KV tokens | Ratio |
|---|---|---|---|---|
| gemma-4-E4B-it | 2 | ~696K | ~1.39M | ~2.00× |
| Qwen2.5-14B | 2 | ~320K | ~640K | ~2.00× |
| Mistral-24B | 2 | ~180K | ~360K | ~2.00× |
| Llama-3.1-70B | 4 | ~90K | ~180K | ~2.00× |
| Qwen2.5-72B | 4 | ~88K | ~176K | ~2.00× |
| DeepSeek-R1-Distill-Llama-70B | 4 | ~90K | ~180K | ~2.00× |

The theoretical ratio is always **2.0×** (8-bit vs 16-bit). Real measurements
may vary slightly due to block alignment and memory fragmentation.

---

## Collecting results

Results are written directly to per-script subdirectories under `/tmp/` inside the container,
which are bind-mounted to the same names under `~/LLM/` on the host. No `docker cp` needed —
results are immediately visible on the host as the benchmark runs.

| Script | Container path | Host path |
|---|---|---|
| `bench_fp8_kv_perf.sh` | `/tmp/fp8kv_perf/` | `~/LLM/fp8kv_perf/` |
| `bench_fp8_kv_sla_concurrency.sh` | `/tmp/fp8kv_sla/` | `~/LLM/fp8kv_sla/` |
| `bench_fp8_kv_long_context.sh` | `/tmp/fp8kv_long_context/` | `~/LLM/fp8kv_long_context/` |
| `bench_fp8_kv_accuracy.sh` | `/tmp/fp8kv_accuracy/` | `~/LLM/fp8kv_accuracy/` |
| `run_fp8_kv_sweep.sh` | `/tmp/fp8_kv_sweep/` | `~/LLM/fp8_kv_sweep/` |

```bash
# List results on the host
ls ~/LLM/fp8kv_perf/
ls ~/LLM/fp8kv_sla/
ls ~/LLM/fp8kv_long_context/
ls ~/LLM/fp8kv_accuracy/
```
