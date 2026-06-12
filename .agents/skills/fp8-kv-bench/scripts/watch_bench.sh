#!/usr/bin/env bash
# watch_bench.sh — background monitor for fp8-kv benchmark runs
#
# Usage:
#   nohup bash watch_bench.sh <csv_path> <expected_rows> <bench_process_name> \
#       [poll_interval_sec] > /home/intel/LLM/watch_<TS>.log 2>&1 &
#
# Arguments:
#   csv_path           Full path to the CSV file being written (on host)
#   expected_rows      Number of data rows expected when complete (excl. header)
#   bench_process_name grep pattern to detect the running bench process
#   poll_interval_sec  Seconds between polls (default: 300)
#
# Example:
#   nohup bash watch_bench.sh \
#       /home/intel/LLM/ruler_summary_20260522_065626.csv \
#       78 \
#       bench_fp8_kv_accuracy \
#       300 > /home/intel/LLM/watch_20260522_065626.log 2>&1 &

set -euo pipefail

CSV_PATH="${1:?Usage: watch_bench.sh <csv_path> <expected_rows> <bench_process_name> [poll_sec]}"
EXPECTED_ROWS="${2:?}"
BENCH_PATTERN="${3:?}"
POLL_SEC="${4:-300}"

LAPTOP_DEST="/Users/kmittal/Desktop/Intel/GTM/kv-cache-quant"
CSV_BASENAME="$(basename "$CSV_PATH")"
VALIDATE_SCRIPT="/home/intel/vllm/.agents/skills/fp8-kv-bench/scripts/validate_csv.py"
LOG_DIR="/home/intel/LLM"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== watch_bench.sh started ==="
log "Watching : $CSV_PATH"
log "Expected : $EXPECTED_ROWS rows"
log "Process  : $BENCH_PATTERN"
log "Poll     : every ${POLL_SEC}s"
echo ""

while true; do
    # Count data rows (subtract header)
    if [[ -f "$CSV_PATH" ]]; then
        total_lines=$(wc -l < "$CSV_PATH")
        rows=$(( total_lines - 1 ))
    else
        rows=0
    fi

    # Check if bench process is still alive inside container
    alive=$(docker exec vllm-test pgrep -c -f "$BENCH_PATTERN" 2>/dev/null || echo 0)

    pct=$(( rows * 100 / (EXPECTED_ROWS > 0 ? EXPECTED_ROWS : 1) ))
    log "Progress: ${rows}/${EXPECTED_ROWS} rows (${pct}%) | process alive: ${alive}"

    # Check for errors in latest server log
    latest_log=$(ls -t "$LOG_DIR"/vllm_ruler_server_*.log 2>/dev/null | head -1 || true)
    if [[ -n "$latest_log" ]]; then
        errs=$(tail -10 "$latest_log" | grep -icE "error|oom|killed|crash" || true)
        if [[ "$errs" -gt 0 ]]; then
            log "WARNING: $errs error/OOM/crash signal(s) in $latest_log"
        fi
    fi

    # --- COMPLETE ---
    if [[ "$alive" -eq 0 && "$rows" -ge "$EXPECTED_ROWS" ]]; then
        echo ""
        log "=== COMPLETE ==="
        log "Run finished with ${rows} rows (expected ${EXPECTED_ROWS})."
        echo ""
        echo "------------------------------------------------------------"
        echo "VALIDATE (run on server):"
        echo "  python3 $VALIDATE_SCRIPT \\"
        echo "      $CSV_PATH \\"
        echo "      $LOG_DIR"
        echo ""
        echo "COPY TO LAPTOP (run on your Mac):"
        echo "  scp intel@b70-server-sc-3:${CSV_PATH} \\"
        echo "      ${LAPTOP_DEST}/"
        echo "------------------------------------------------------------"
        exit 0
    fi

    # --- CRASHED ---
    if [[ "$alive" -eq 0 && "$rows" -lt "$EXPECTED_ROWS" ]]; then
        echo ""
        log "=== CRASHED / STOPPED ==="
        log "Process gone but only ${rows}/${EXPECTED_ROWS} rows written."
        echo ""
        echo "------------------------------------------------------------"
        echo "LAST 30 LINES OF BENCH LOG:"
        bench_log=$(ls -t /tmp/bench_*_accuracy_*.log /tmp/bench_*_perf_*.log \
                       /tmp/bench_*_sla_*.log /tmp/bench_*_long_*.log 2>/dev/null | head -1 || true)
        if [[ -n "$bench_log" ]]; then
            docker exec vllm-test tail -30 "$bench_log" 2>/dev/null || tail -30 "$bench_log" 2>/dev/null || echo "(log not found)"
        else
            echo "(no bench log found)"
        fi
        echo "------------------------------------------------------------"
        echo "Consider running the backfill procedure (see SKILL.md)."
        exit 1
    fi

    sleep "$POLL_SEC"
done
