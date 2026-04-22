#!/usr/bin/env bash
# Step 3 (gpudrive env): extract the full 2984-dim Waymo observation for every
# tfrecord-*.json produced by Step 2.
#
# Assumes the caller has already `conda activate gpudrive` (or that the
# gpudrive env's python is on PATH).
set -euo pipefail

STEP3=/data/llh/navsim_workspace/navsim/navsim/planning/script/step3_extract_road_obs.py

python "$STEP3" --auto --isolated --isolated_batch_size 64 --parallel_workers 10
