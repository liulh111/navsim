# import debugpy
# debugpy.listen(("localhost", 9501))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

import os
import logging
import numpy as np
import hydra
from hydra.utils import instantiate

from pathlib import Path
from omegaconf import DictConfig
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.enums import BoundingBoxIndex
from navsim.common.dataloader import MetricCacheLoader, SceneLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navsim.visualization.plots import (
    configure_bev_ax,
    plot_cameras_frame,
    plot_cameras_frame_with_annotations,
    plot_cameras_frame_with_lidar,
    frame_plot_to_gif,
)
from navsim.visualization.bev import add_annotations_to_bev_ax, add_lidar_to_bev_ax


from navsim.planning.script.visualization import parse_observation, visualize_single_obs_and_save


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


# ── Constants ──────────────────────────────────────────────────────────────────
PARTNER_NUM = 63
PARTNER_DIM = 6
ROAD_NUM = 200
ROAD_DIM = 13
EGO_DIM = 6
TOTAL_DIM = EGO_DIM + PARTNER_NUM * PARTNER_DIM + ROAD_NUM * ROAD_DIM  # 2984


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def extract_waymo_whitebox_features(scene: Scene) -> np.ndarray:
    """
    Extract Waymo-format whitebox features from a NAVSIM Scene.
    Returns a flat float32 vector of length TOTAL_DIM (2984).
    """
    frame_idx = scene.scene_metadata.num_history_frames - 1
    ego_status = scene.frames[frame_idx].ego_status
    annotations = scene.frames[frame_idx].annotations

    # ── 1. ego-LocalEgoState (1 x 6) ──────────────────────────────────────
    speed = float(np.linalg.norm(ego_status.ego_velocity))

    # Fixed Pacifica parameters (navsim uses a single vehicle type)
    vehicle_params = get_pacifica_parameters()
    vehicle_length = vehicle_params.length
    vehicle_width = vehicle_params.width

    # rel_goal: GT trajectory endpoint in ego-local frame
    trajectory = scene.get_future_trajectory(
        num_trajectory_frames=scene.scene_metadata.num_future_frames
    )
    rel_goal_x = float(trajectory.poses[-1, 0])
    rel_goal_y = float(trajectory.poses[-1, 1])

    # is_collided: not available in open-loop, hardcode 0
    is_collided = 0.0

    ego_state = np.array(
        [speed, vehicle_length, vehicle_width, rel_goal_x, rel_goal_y, is_collided],
        dtype=np.float32,
    )

    # ── 2. partner_state (63 x 6) ─────────────────────────────────────────
    partner_state = _extract_partner_state(annotations)

    # ── 3. road_state (200 x 13) ──────────────────────────────────────────
    road_state = np.zeros((ROAD_NUM, ROAD_DIM), dtype=np.float32)  # placeholder for now

    # ── Flatten and concatenate ────────────────────────────────────────────
    features = np.concatenate([
        ego_state,                    # (6,)
        partner_state.flatten(),      # (378,)
        road_state.flatten(),         # (2600,)
    ]).astype(np.float32)

    assert features.shape == (TOTAL_DIM,), f"Expected {TOTAL_DIM}, got {features.shape}"
    return features

def _extract_partner_state(annotations) -> np.ndarray:
    N = annotations.boxes.shape[0]
    if N == 0:
        return np.zeros((PARTNER_NUM, PARTNER_DIM), dtype=np.float32)

    speeds = np.linalg.norm(annotations.velocity_3d[:, :2], axis=1)

    rel_pos_x = annotations.boxes[:, BoundingBoxIndex.X]
    rel_pos_y = annotations.boxes[:, BoundingBoxIndex.Y]

    orientation = annotations.boxes[:, BoundingBoxIndex.HEADING]
    orientation = _wrap_to_pi(orientation)

    vehicle_length = annotations.boxes[:, BoundingBoxIndex.LENGTH]
    vehicle_width  = annotations.boxes[:, BoundingBoxIndex.WIDTH]

    all_partners = np.stack(
        [speeds, rel_pos_x, rel_pos_y, orientation, vehicle_length, vehicle_width],
        axis=1,
    ).astype(np.float32)

    distances = np.sqrt(rel_pos_x ** 2 + rel_pos_y ** 2)
    sorted_indices = np.argsort(distances)
    all_partners = all_partners[sorted_indices]

    result = np.zeros((PARTNER_NUM, PARTNER_DIM), dtype=np.float32)
    count = min(N, PARTNER_NUM)
    result[:count] = all_partners[:count]
    return result

def navsim_self_visualization(scene: Scene, token: str) -> None:

    VIZ_DIR = Path(os.path.join("viz_output", token))
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # 取 scene 中历史帧的最后一帧（即当前时刻）的索引
    current_frame_idx = scene.scene_metadata.num_history_frames - 1

    # --- 1. 仅标注框的自定义 BEV ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_title(f"BEV Annotations  |  token: {token[:12]}...")
    add_annotations_to_bev_ax(ax, scene.frames[current_frame_idx].annotations)
    configure_bev_ax(ax)
    out_path = VIZ_DIR / f"bev_annotations.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved BEV annotations -> {out_path}")

    # --- 2. 相机 3×3 网格图（中心为 BEV）---
    fig, ax = plot_cameras_frame(scene, current_frame_idx)
    out_path = VIZ_DIR / f"cameras.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cameras frame -> {out_path}")

    # --- 3. 相机图 + bounding-box 标注 ---
    fig, ax = plot_cameras_frame_with_annotations(scene, current_frame_idx)
    out_path = VIZ_DIR / f"cameras_annotations.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cameras+annotations -> {out_path}")

    # --- 4. 相机图 + LiDAR 点云投影 ---
    fig, ax = plot_cameras_frame_with_lidar(scene, current_frame_idx)
    out_path = VIZ_DIR / f"cameras_lidar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cameras+lidar -> {out_path}")

    # --- 5. BEV + LiDAR 自定义组合图 ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_title(f"BEV LiDAR+Annotations  |  token: {token[:12]}...")
    add_annotations_to_bev_ax(ax, scene.frames[current_frame_idx].annotations)
    add_lidar_to_bev_ax(ax, scene.frames[current_frame_idx].lidar)
    configure_bev_ax(ax)
    out_path = VIZ_DIR / f"bev_lidar_annotations.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved BEV lidar+annotations -> {out_path}")

    # --- 6. 所有帧的动态 GIF（以相机+标注为例）---
    frame_indices = list(range(len(scene.frames)))
    gif_path = str(VIZ_DIR / f"cameras_annotations.gif")
    frame_plot_to_gif(
        gif_path,
        plot_cameras_frame_with_annotations,
        scene,
        frame_indices,
    )
    logger.info(f"Saved GIF -> {gif_path}")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        # original_sensor_path=None,
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        # sensor_config=SensorConfig.build_no_sensors(),
        sensor_config=SensorConfig.build_all_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_loader_tokens_stage_one = scene_loader.tokens_stage_one
    tokens_to_evaluate_stage_one = sorted(set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens))
    if not tokens_to_evaluate_stage_one:
        logger.warning("No stage-one tokens overlap between SceneLoader and MetricCacheLoader.")
        return

    token = tokens_to_evaluate_stage_one[0]
    scene = scene_loader.get_scene_from_token(token)
    
    navsim_self_visualization(scene, token)

    VIZ_DIR = Path(os.path.join("viz_output", token))
    save_path = VIZ_DIR / f"waymo_whitebox_obs.png"

    obs_2984 = extract_waymo_whitebox_features(scene)
    visualize_single_obs_and_save(obs_2984=obs_2984, save_path=str(save_path))

if __name__ == "__main__":
    main()
