"""
Step 1: Export rich scene metadata from navsim (run in navsim-llh env).

Selects one token from stage-1 and one from stage-2 of navhard_two_stage
(intersected with the metric cache), and exports a pickle per token that
carries everything downstream stages need to produce the Waymo 2984-dim
observation via ScenarioMax + GPUDrive:

    - token, stage, map_name
    - log_name, frame_token, timestamp
    - ego_pose (global [x, y, heading])
    - ego_velocity_local [vx, vy]
    - ego_size [length, width, height]
    - goal_local [gx, gy]         (stage-1: future traj endpoint; stage-2: (0,0))
    - partners_local: list of {annotation_index, rel_x, rel_y, rel_z, heading,
                               vel_x, vel_y,
                               length, width, height, name,
                               instance_token, track_token}
                      — in annotation order (no distance sort, to match the
                        standard ScenarioMax → nuPlan → GPUDrive convention)

Usage:
    conda activate navsim-llh
    cd $NAVSIM_DEVKIT_ROOT
    bash scripts/evaluation/run_step1_export_metadata.sh
"""
import logging
import pickle
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.enums import BoundingBoxIndex

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

OUTPUT_BASE = Path("/data/llh/navsim_workspace/exp/pipeline_output")

DEFAULT_EGO_HEIGHT = 1.777  # Pacifica height used by nuPlan / ScenarioMax


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _export_one_token(scene: Scene, token: str, stage: str) -> Path:
    """Serialise the rich metadata for a single scene to a pickle."""
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]

    ego_pose = frame.ego_status.ego_pose  # [x, y, heading], global
    ego_velocity_local = frame.ego_status.ego_velocity  # [vx, vy], ego-local

    pacifica = get_pacifica_parameters()
    ego_size = [
        float(pacifica.length),
        float(pacifica.width),
        float(getattr(pacifica, "height", DEFAULT_EGO_HEIGHT)),
    ]

    # Goal: stage-1 uses future trajectory endpoint; stage-2 falls back to (0, 0)
    goal_local = [0.0, 0.0]
    if stage == "stage_one":
        trajectory = scene.get_future_trajectory(
            num_trajectory_frames=scene.scene_metadata.num_future_frames
        )
        goal_local = [float(trajectory.poses[-1, 0]), float(trajectory.poses[-1, 1])]
    else:
        try:
            trajectory = scene.get_future_trajectory(
                num_trajectory_frames=scene.scene_metadata.num_future_frames
            )
            if hasattr(trajectory, "poses") and len(trajectory.poses) > 0:
                goal_local = [
                    float(trajectory.poses[-1, 0]),
                    float(trajectory.poses[-1, 1]),
                ]
        except Exception as exc:
            logger.warning(
                "[%s] future trajectory unavailable for %s, goal=(0,0): %s",
                stage, token, exc,
            )

    # Partners in annotation order (no distance sort)
    annotations = frame.annotations
    partners_local = []
    if annotations is not None and annotations.boxes.shape[0] > 0:
        boxes = annotations.boxes
        vel3d = annotations.velocity_3d
        names = list(annotations.names) if hasattr(annotations, "names") else []
        instance_tokens = (
            list(annotations.instance_tokens)
            if hasattr(annotations, "instance_tokens")
            else []
        )
        track_tokens = (
            list(annotations.track_tokens)
            if hasattr(annotations, "track_tokens")
            else []
        )
        for i in range(boxes.shape[0]):
            partners_local.append(
                {
                    "annotation_index": int(i),
                    "rel_x": float(boxes[i, BoundingBoxIndex.X]),
                    "rel_y": float(boxes[i, BoundingBoxIndex.Y]),
                    "rel_z": float(boxes[i, BoundingBoxIndex.Z]),
                    "heading": _wrap_to_pi(float(boxes[i, BoundingBoxIndex.HEADING])),
                    "vel_x": float(vel3d[i, 0]),
                    "vel_y": float(vel3d[i, 1]),
                    "length": float(boxes[i, BoundingBoxIndex.LENGTH]),
                    "width": float(boxes[i, BoundingBoxIndex.WIDTH]),
                    "height": float(boxes[i, BoundingBoxIndex.HEIGHT]),
                    "name": str(names[i]) if i < len(names) else "",
                    "instance_token": (
                        str(instance_tokens[i]) if i < len(instance_tokens) else ""
                    ),
                    "track_token": (
                        str(track_tokens[i]) if i < len(track_tokens) else ""
                    ),
                }
            )

    metadata = {
        "token": token,
        "stage": stage,
        "log_name": scene.scene_metadata.log_name,
        "scene_token": scene.scene_metadata.scene_token,
        "frame_token": frame.token,
        "current_frame_index": int(frame_idx),
        "timestamp": int(frame.timestamp),
        "map_name": scene.scene_metadata.map_name,
        "ego_pose": [float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])],
        "ego_velocity_local": [float(ego_velocity_local[0]), float(ego_velocity_local[1])],
        "ego_size": ego_size,
        "goal_local": goal_local,
        "partners_local": partners_local,
    }

    output_dir = OUTPUT_BASE / token
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scene_metadata.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(
        "[%s] token=%s map=%s ego_pose=%s partners=%d goal_local=%s -> %s",
        stage, token, metadata["map_name"], metadata["ego_pose"],
        len(partners_local), metadata["goal_local"], output_path,
    )
    return output_path


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    # `export_all` is an optional Hydra override (default False → only the
    # first available token per stage is exported, matching the original
    # smoke-test workflow). Pass `+export_all=true` to iterate every token
    # intersected with the metric cache in both stages.
    export_all = bool(cfg.get("export_all", False))

    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    metric_tokens = set(metric_cache_loader.tokens)

    stage_plans = [
        ("stage_one", sorted(set(scene_loader.tokens_stage_one) & metric_tokens)),
        ("stage_two", sorted(set(scene_loader.reactive_tokens_stage_two) & metric_tokens)),
    ]

    for stage, tokens in stage_plans:
        if not tokens:
            logger.error("No %s tokens found in the intersection with metric cache.", stage)
            continue

        selected = tokens if export_all else tokens[:1]
        logger.info(
            "[%s] exporting %d / %d tokens (export_all=%s)",
            stage, len(selected), len(tokens), export_all,
        )

        for i, token in enumerate(selected):
            logger.info("[%s] (%d/%d) %s", stage, i + 1, len(selected), token)
            scene = scene_loader.get_scene_from_token(token)
            _export_one_token(scene, token, stage)


if __name__ == "__main__":
    main()
