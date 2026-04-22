#!/usr/bin/env bash
# Evaluate WaymoMLPAgent on navhard_two_stage.
# Assumes the 3-stage pipeline (step1/step2/step3) has already been run for
# every token in both stages — the agent loads the 2984-dim .npy keyed by
# scene token from /data/llh/navsim_workspace/exp/pipeline_output/<TOKEN>/.
set -euo pipefail

export RAY_TMPDIR=/data/llh/ray_tmp
mkdir -p $RAY_TMPDIR

TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=waymo_mlp_agent \
    experiment_name=waymo_mlp_agent \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH
