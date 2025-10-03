#!/usr/bin/env bash
# run_benchmark_iterations.sh
# ---------------------------------------------------------------------
# Run N iterations of full model training and save benchmark results
# ---------------------------------------------------------------------

set -euo pipefail

TOTAL_ITERATIONS=30        # adjust as needed
RUN_SCRIPT="./run_all_models.sh"
LOG_DIR="./logs"
SUMMARY_CSV="./benchmark_timing_summary.csv"

# Ensure logs folder exists
mkdir -p "$LOG_DIR"

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "❌ Error: $RUN_SCRIPT not found!"
  exit 1
fi

# Detect where to resume (safe: won't fail if no files exist)
existing_iters=($(find "$LOG_DIR" -maxdepth 1 -name 'iter*.out' \
  | sed -E 's/.*iter([0-9]+)\.out/\1/' | sort -n))

if [ ${#existing_iters[@]} -gt 0 ]; then
  last_completed=${existing_iters[-1]}
  START_ITER=$((last_completed + 1))
else
  echo "⚠️ No existing iteration logs found. Starting from iteration 1."
  START_ITER=1
fi

# Initialize timing summary if missing
if [ ! -f "$SUMMARY_CSV" ]; then
  echo "iteration,start_time,end_time,duration_sec,gpu_id" > "$SUMMARY_CSV"
fi

echo "🔎 Resuming from iteration $START_ITER"

# GPU ID detection
if command -v nvidia-smi &> /dev/null; then
  GPU_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -n1)
else
  GPU_ID=0
fi

run_iteration() {
  local i=$1
  local tag="iter${i}"
  local log_file="${LOG_DIR}/${tag}.out"
  local start_time=$(date +%s)

  echo "🚀 [$tag] Starting on GPU $GPU_ID at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" METRIC_PREFIX="$tag" SEED_OFFSET="$i" \
      bash "$RUN_SCRIPT" > "$log_file" 2>&1

  local end_time=$(date +%s)
  local duration=$((end_time - start_time))

  echo "$tag,$(date -d @$start_time +'%Y-%m-%d %H:%M:%S'),$(date -d @$end_time +'%Y-%m-%d %H:%M:%S'),$duration,$GPU_ID" >> "$SUMMARY_CSV"
  echo "✅ [$tag] Done in ${duration}s"
}

if (( START_ITER > TOTAL_ITERATIONS )); then
  echo "✅ All $TOTAL_ITERATIONS iterations already completed."
  exit 0
fi

# Start iterations
for ((i = START_ITER; i <= TOTAL_ITERATIONS; i++)); do
  run_iteration "$i"
  sleep 2  # Small pause to reduce load on filesystem
  sync     # Ensure filesystem buffers are flushed
  echo "🚀 Finished iteration $i."
done

# Merge results after iterations complete
if find "$LOG_DIR" -maxdepth 1 -name 'iter*.out' | grep -q .; then
  echo "=== Merging iteration results ==="
  python3 merge_benchmark_results.py
  python3 wilcoxon_test.py
  python3 plot_f1_distributions.py
  python3 summarize_benchmark.py
  echo "🎉 Benchmarking complete! All outputs updated."
else
  echo "⚠️ No iteration outputs found to merge. Skipping post-processing."
fi
