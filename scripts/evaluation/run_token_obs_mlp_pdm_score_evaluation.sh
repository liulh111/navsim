TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
TOKEN_OBS_DIR="/data/llh/navsim_workspace/results/gpudrive_json/navhard_two_stage_npy"
MLP_MODEL_MODULE_PATH="/data/llh/navsim_workspace/navsim/scripts/evaluation/simple_token_obs_mlp.py"
MLP_MODEL_CLASS_NAME="SimpleTokenObsMLP"
MLP_MODEL_CONFIG_PATH=/data/llh/navsim_workspace/results/token_obs_mlp_demo/config.json
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp}

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_token_obs_mlp_pdm_score.py" \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=token_obs_mlp_agent \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
agent.obs_dir="$TOKEN_OBS_DIR" \
agent.model_module_path="$MLP_MODEL_MODULE_PATH" \
agent.model_class_name="$MLP_MODEL_CLASS_NAME" \
agent.model_config_path="$MLP_MODEL_CONFIG_PATH"
