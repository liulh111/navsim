第一步：NAVSIM 环境

conda activate navsim-llh
cd /data/llh/navsim_workspace

PYTHONPATH=$PWD/navsim \
python navsim/scripts/export_navhard_stage_one_index.py

第二步：ScenarioMax 环境

conda activate navsim
cd /data/llh/navsim_workspace/ScenarioMax

python scripts/export_navhard_stage_one_gpudrive_json.py \
--index ../results/gpudrive_json/navhard_two_stage/stage_one_index.csv \
--output-dir ../results/gpudrive_json/navhard_two_stage/stage_one \
--nuplan-maps-root ../dataset/maps

第三步：GPUDrive中rollout
conda run -n gpudrive python navsim/scripts/run_navhard_stage_one_gpudrive_rollout.py

第四步：评估结果

conda run -n navsim-llm python \
navsim/navsim/planning/script/run_pdm_score_one_stage.py \
--config-name default_run_offline_stage_one_pdm_score \
agent.trajectory_dir=/data/llh/navsim_workspace/results/gpudrive_rollouts/navhard_stage_one/expert_bicycle/trajectories