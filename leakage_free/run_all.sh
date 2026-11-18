#!/usr/bin/env bash
# Re-runs every leakage-free experiment, then rebuilds the summary tables.
# CPU-only; each script runs single-threaded, so the four governorates are run in parallel.
# Total runtime on a 4-core CPU: roughly 6-10 hours (bench.py rule A and mc.py are the slowest).
set -euo pipefail
cd "$(dirname "$0")/src"
REGIONS="Erbil Duhok Sulaymaniyah Halabja"
run() { for r in $REGIONS; do python "$@" "$r" & done; wait; }

python leakage_check.py                         # original vs. clean protocol, BiLSTM
for r in $REGIONS; do python bench.py "$r" & done; wait                     # rule A
for r in $REGIONS; do python bench.py "$r" --fixed 100 & done; wait        # rule B
for r in $REGIONS; do python bench.py "$r" --fixed 100 --baselines & done; wait
run logo.py
run ablation.py
run gapfill.py
run mc.py
run scenarios.py
python summarize.py
