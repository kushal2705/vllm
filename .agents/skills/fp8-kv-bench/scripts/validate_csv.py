#!/usr/bin/env python3
"""
validate_csv.py — Cross-check ruler_summary CSV against raw eval log files.

Usage:
    python3 validate_csv.py <csv_file> <log_dir>

Exit 0 if all rows match; exit 1 if any discrepancy.
"""
import csv, json, re, os, sys
from collections import defaultdict


def parse_log(path):
    """Find the last complete JSON object in an eval log (the results.json dump)."""
    text = open(path).read()
    pos = text.rfind('\n{')
    if pos == -1:
        pos = text.rfind('{')
    try:
        return json.loads(text[pos:].strip())
    except Exception:
        return None


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <csv_file> <log_dir>")
        sys.exit(2)

    csv_file = sys.argv[1]
    log_dir  = sys.argv[2]

    TP_MAP = {
        'llama31': 1, 'deepseekr1': 1, 'gemma3': 1, 'qwen3': 1,
        'gemma4': 2, 'qwen25': 2, 'mistral': 2,
        'llama33_70b': 4, 'qwen25_72b': 4, 'deepseekr1_70b': 4,
    }

    # ── Load log files ────────────────────────────────────────────────────────
    log_data = {}
    pattern = re.compile(r'ruler_(\w+)_(bf16|fp8)_ctx(\d+)\.log$')
    for fname in sorted(os.listdir(log_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        model, config, ctx_len = m.group(1), m.group(2), int(m.group(3))
        raw = parse_log(os.path.join(log_dir, fname))
        if raw is None:
            print(f"WARN: could not parse {fname}", file=sys.stderr)
            continue
        log_data[(model, config, ctx_len)] = raw

    print(f"Parsed {len(log_data)} log files")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_rows = {}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            key = (row['model'], row['config'], int(row['context_length']), row['task'])
            csv_rows[key] = float(row['score'])

    print(f"CSV rows loaded: {len(csv_rows)}")

    # ── Cross-check ───────────────────────────────────────────────────────────
    missing, wrong, extra = [], [], []

    for (model, config, ctx_len), tasks in log_data.items():
        for task_name, metrics in tasks.items():
            if task_name == 'alias':
                continue
            for key, val in metrics.items():
                if not isinstance(key, str) or '_stderr' in key or key == 'alias':
                    continue
                ctx_str = key.split(',')[0].strip()
                if not ctx_str.isdigit():
                    continue
                if not isinstance(val, (int, float)) or val < 0:
                    continue
                log_ctx   = int(ctx_str)
                log_score = round(float(val), 4)
                csv_key   = (model, config, log_ctx, task_name)
                if csv_key not in csv_rows:
                    missing.append((csv_key, log_score))
                elif abs(csv_rows[csv_key] - log_score) > 1e-4:
                    wrong.append((csv_key, log_score, csv_rows[csv_key]))

    # Check for CSV rows with no matching log source
    for (model, config, ctx, task), score in csv_rows.items():
        found = any(
            t == task and
            any(
                isinstance(k, str) and not ('_stderr' in k or k == 'alias') and
                k.split(',')[0].strip().isdigit() and
                int(k.split(',')[0].strip()) == ctx and
                isinstance(v, (int, float)) and v >= 0
                for k, v in metrics.items()
            )
            for (lm, lc, lctx), tasks in log_data.items()
            if lm == model and lc == config
            for t, metrics in tasks.items()
            if t == task
        )
        if not found:
            extra.append((model, config, ctx, task, score))

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n=== Validation Results ===")
    print(f"Missing in CSV (log → CSV):  {len(missing)}")
    print(f"Wrong value (log ≠ CSV):     {len(wrong)}")
    print(f"Extra in CSV (no log source): {len(extra)}")

    if missing:
        print("\nMISSING:")
        for k, v in missing[:20]:
            print(f"  {k} → log={v:.4f}")
    if wrong:
        print("\nWRONG VALUES:")
        for k, lv, cv in wrong[:20]:
            print(f"  {k}: log={lv:.4f}  csv={cv:.4f}  diff={cv-lv:+.5f}")
    if extra:
        print("\nEXTRA (spurious CSV rows):")
        for row in extra[:20]:
            print(f"  {row}")

    if not missing and not wrong and not extra:
        print("\nALL OK — CSV matches log files exactly.")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
