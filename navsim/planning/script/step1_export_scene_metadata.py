"""
Step 1: Export scene metadata from navsim (run in navsim-llh env).

Loads one scene from navhard_two_stage stage1, exports
{token, map_name, ego_pose} to a pickle file.

Usage:
    conda activate navsim-llh
    cd $NAVSIM_DEVKIT_ROOT
    bash scripts/evaluation/run_step1_export_metadata.sh
"""
import logging
import pickle

import hydra
from hydra.utils import instantiate
from pathlib import Path
from omegaconf import DictConfig

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

# All outputs go under this absolute path to avoid hydra cwd issues
OUTPUT_BASE = Path("/data/llh/navsim_workspace/exp/pipeline_output")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:

    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens = sorted(
        set(scene_loader.tokens_stage_one) & set(metric_cache_loader.tokens)
    )
    if not tokens:
        logger.error("No stage-one tokens found.")
        return

    token = tokens[0]
    scene = scene_loader.get_scene_from_token(token)

    frame_idx = scene.scene_metadata.num_history_frames - 1
    ego_pose = scene.frames[frame_idx].ego_status.ego_pose  # [x, y, heading]

    metadata = {
        "token": token,
        "map_name": scene.scene_metadata.map_name,
        "ego_pose": ego_pose.tolist(),  # list for portability
    }

    output_dir = OUTPUT_BASE / token
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scene_metadata.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Token:    {token}")
    logger.info(f"Map:      {metadata['map_name']}")
    logger.info(f"Ego pose: {metadata['ego_pose']}")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
