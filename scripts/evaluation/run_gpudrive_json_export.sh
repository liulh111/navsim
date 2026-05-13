TRAIN_TEST_SPLIT=navhard_two_stage
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
RESULTS_ROOT=${NAVSIM_RESULTS_ROOT:-${NAVSIM_DEVKIT_ROOT}/../results}
GPUDRIVE_JSON_OUTPUT_DIR=${RESULTS_ROOT}/gpudrive_json/${TRAIN_TEST_SPLIT}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp}

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_gpudrive_json_export.py \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=gpudrive_json_export \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
gpudrive_json_output_dir=$GPUDRIVE_JSON_OUTPUT_DIR \
reference_json_dir=auto \
worker=ray_distributed_no_torch
