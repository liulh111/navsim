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

第一，二步：直接出GPUDrive json
conda run -n navsim-llh python navsim/navsim/planning/script/run_gpudrive_json_export.py \
export_stage=all \
gpudrive_json_output_dir=/data/llh/navsim_workspace/results/gpudrive_json/navhard_two_stage \
separate_stage_dirs=true \
overwrite=true

第三步：GPUDrive中rollout
conda run -n gpudrive python navsim/scripts/run_navhard_stage_one_gpudrive_rollout.py \
--stage all \
--json-dir /data/llh/navsim_workspace/results/gpudrive_json/navhard_two_stage \
--output-dir /data/llh/navsim_workspace/results/gpudrive_rollouts/navhard_two_stage \
--batch-size 16 \
--ego-policy zero \
--device cpu \
--overwrite

第四步：评估结果

1. 评测 Stage One

conda run -n navsim python navsim/navsim/planning/script/run_gpudrive_two_stage_pdm_score.py \
evaluation_stage=stage_one \
stage_one_trajectory_dir=/data/llh/navsim_workspace/results/gpudrive_rollouts/navhard_two_stage/stage_one/zero_idm/trajectories \
output_dir=/data/llh/navsim_workspace/results/pdm_score/gpudrive_zero_idm_stage_one

2. 评测 Two-Stage

conda run -n navsim python navsim/navsim/planning/script/run_gpudrive_two_stage_pdm_score.py \
evaluation_stage=all \
stage_one_trajectory_dir=/data/llh/navsim_workspace/results/gpudrive_rollouts/navhard_two_stage/stage_one/zero_idm/trajectories \
stage_two_trajectory_dir=/data/llh/navsim_workspace/results/gpudrive_rollouts/navhard_two_stage/stage_two/zero_idm/trajectories \
output_dir=/data/llh/navsim_workspace/results/pdm_score/gpudrive_zero_idm_two_stage

3. 可视化 Stage One 评测结果

把 <CSV> 换成上面 Stage One 评测输出目录里的 csv：

conda run -n navsim python scripts/evaluation/visualize_pdm_rollout_cases.py \
--stage stage1 \
--quality worst \
--num_scenes 20 \
--csv_path /data/llh/navsim_workspace/results/pdm_score/gpudrive_zero_idm_stage_one/<CSV> \
--sort_by score \
--output_dir /data/llh/navsim_workspace/results/visualization/gpudrive_zero_idm_stage_one_worst \
--config_name default_run_pdm_score \
--agent offline_trajectory_agent \
--override agent.trajectory_dir=/data/llh/navsim_workspace/results/gpudrive_rollouts/navhard_two_stage/stage_one/zero_idm/trajectories

如果你用的是 expert_bicycle_idm，把所有 zero_idm 路径替换成：

expert_bicycle_idm
