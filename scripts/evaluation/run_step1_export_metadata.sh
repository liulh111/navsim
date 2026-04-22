#!/usr/bin/env bash
# Step 1 (navsim-llh env): export scene metadata for navhard_two_stage.
#
# Default: exports only the first token of each stage (quick smoke test).
# Pass "all" as the first argument to export every token in both stages.
#
# Usage:
#   bash scripts/evaluation/run_step1_export_metadata.sh          # first token per stage
#   bash scripts/evaluation/run_step1_export_metadata.sh all      # all tokens in both stages
set -euo pipefail

TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

EXPORT_ALL_ARG=""
if [[ "${1:-}" == "all" ]]; then
    EXPORT_ALL_ARG="+export_all=true"
fi

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/step1_export_scene_metadata.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=step1_export_metadata \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
    worker=sequential \
    $EXPORT_ALL_ARG
