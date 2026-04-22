#!/usr/bin/env bash
# Step 2 (ScenarioMax .venv): build full GPUDrive JSON (ego + partners + roads)
# for every scene_metadata.pkl produced by Step 1.
#
# Run from any shell; does NOT require `conda activate`.
set -euo pipefail

SCENARIOMAX_PY=/data/llh/navsim_workspace/ScenarioMax/.venv/bin/python
STEP2=/data/llh/navsim_workspace/navsim/navsim/planning/script/step2_generate_gpudrive_json.py

"$SCENARIOMAX_PY" "$STEP2" --auto
