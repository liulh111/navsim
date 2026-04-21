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

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType

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


from navsim.planning.script.visualization import visualize_single_obs_and_save


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


SEARCH_RADIUS = 200.0  # meters for map query
MAX_POINTS_PER_TYPE = 5000  # max road points per type (closest by distance)

# Waymo road type indices: 0=none, 1=RoadLine, 2=RoadEdge, 3=RoadLane, 4=CrossWalk, 5=SpeedBump, 6=StopSign
ROAD_TYPE_NAMES = ["none", "RoadLine", "RoadEdge", "RoadLane", "CrossWalk", "SpeedBump", "StopSign"]

# Which road types to extract. Comment out any to disable.
# Note: SpeedBump is not available in nuPlan map data.
ENABLED_ROAD_TYPES = [
    "RoadLine",    # 1 — lane boundary lines (between adjacent lanes)
    "RoadEdge",    # 2 — road edge boundaries (no adjacent lane)
    "RoadLane",    # 3 — lane center lines (LANE + LANE_CONNECTOR baseline paths)
    "CrossWalk",   # 4 — crosswalk polygons
    # "SpeedBump", # 5 — not available in nuPlan
    "StopSign",    # 6 — stop sign polygons (from STOP_LINE)
]


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _polyline_to_road_points(discrete_path, type_index: int, width: float = 0.0, height: float = 0.0):
    """Convert a polyline (list of StateSE2) to road point rows.

    Each consecutive pair of points becomes one observation, positioned at the
    segment midpoint (matching GPUDrive's makeRoadEdge).

    Returns list of [mid_x, mid_y, half_length, width, height, orientation, type_index].
    """
    n = len(discrete_path)
    if n < 2:
        return []
    points = []
    for i in range(n - 1):
        p1 = discrete_path[i]
        p2 = discrete_path[i + 1]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        half_len = np.sqrt(dx ** 2 + dy ** 2) / 2.0
        orientation = np.arctan2(dy, dx)
        mid_x = (p1.x + p2.x) / 2.0
        mid_y = (p1.y + p2.y) / 2.0
        points.append([mid_x, mid_y, half_len, width, height, orientation, type_index])
    return points


def _polygon_to_road_points(polygon, type_index: int, width: float = 0.0, height: float = 0.0):
    """Convert a shapely Polygon exterior ring to road point rows, using segment midpoints."""
    coords = list(polygon.exterior.coords)
    n = len(coords) - 1  # last coord == first for closed ring
    if n < 2:
        return []
    points = []
    for i in range(n):
        x1, y1 = coords[i][0], coords[i][1]
        x2, y2 = coords[i + 1][0], coords[i + 1][1]
        dx = x2 - x1
        dy = y2 - y1
        half_len = np.sqrt(dx ** 2 + dy ** 2) / 2.0
        orientation = np.arctan2(dy, dx)
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        points.append([mid_x, mid_y, half_len, width, height, orientation, type_index])
    return points


def _extract_road_state(scene: Scene, frame_idx: int,
                        enabled_types: list = None) -> np.ndarray:
    """Extract road state features (N x 13) from nuPlan map API.

    Args:
        enabled_types: list of type names to extract, e.g. ["RoadLine", "RoadEdge"].
                       Defaults to ENABLED_ROAD_TYPES.
    """
    if scene.map_api is None:
        return np.zeros((0, ROAD_DIM), dtype=np.float32)

    if enabled_types is None:
        enabled_types = ENABLED_ROAD_TYPES
    enabled_set = set(enabled_types)

    ego_pose = scene.frames[frame_idx].ego_status.ego_pose
    ego_x, ego_y, ego_heading = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])

    # Rotation matrix for global-to-ego-local transform
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

    # Determine which map layers to query based on enabled types
    layers = []
    need_lane = "RoadLine" in enabled_set or "RoadEdge" in enabled_set or "RoadLane" in enabled_set
    need_lane_connector = "RoadLane" in enabled_set
    if need_lane:
        layers.append(SemanticMapLayer.LANE)
    if need_lane_connector:
        layers.append(SemanticMapLayer.LANE_CONNECTOR)
    if "CrossWalk" in enabled_set:
        layers.append(SemanticMapLayer.CROSSWALK)
    if "StopSign" in enabled_set:
        layers.append(SemanticMapLayer.STOP_LINE)

    if not layers:
        return np.zeros((0, ROAD_DIM), dtype=np.float32)

    map_objects = scene.map_api.get_proximal_map_objects(
        Point2D(ego_x, ego_y), SEARCH_RADIUS, layers
    )

    all_road_points = []
    seen_boundary_ids = set()

    # --- RoadEdge (type 2) and RoadLine (type 1) from LANE boundaries ---
    # --- RoadLane (type 3) from LANE baseline paths ---
    if need_lane:
        for lane_obj in map_objects.get(SemanticMapLayer.LANE, []):
            # Extract boundaries as RoadEdge / RoadLine
            if "RoadEdge" in enabled_set or "RoadLine" in enabled_set:
                try:
                    left_boundary = lane_obj.left_boundary
                    right_boundary = lane_obj.right_boundary
                except Exception:
                    left_boundary = right_boundary = None

                if left_boundary is not None:
                    try:
                        adjacent = lane_obj.adjacent_edges
                    except Exception:
                        adjacent = (None, None)

                    for side_idx, boundary in enumerate([left_boundary, right_boundary]):
                        if boundary.id in seen_boundary_ids:
                            continue
                        seen_boundary_ids.add(boundary.id)

                        try:
                            b_path = boundary.discrete_path
                        except Exception:
                            continue

                        if adjacent[side_idx] is None:
                            if "RoadEdge" in enabled_set:
                                all_road_points.extend(
                                    _polyline_to_road_points(b_path, type_index=2, width=0.15, height=0.15)
                                )
                        else:
                            if "RoadLine" in enabled_set:
                                all_road_points.extend(
                                    _polyline_to_road_points(b_path, type_index=1, width=0.1, height=0.0)
                                )

            # Extract baseline path as RoadLane
            if "RoadLane" in enabled_set:
                try:
                    baseline = lane_obj.baseline_path.discrete_path
                    all_road_points.extend(
                        _polyline_to_road_points(baseline, type_index=3, width=0.0, height=0.0)
                    )
                except Exception:
                    pass

    # --- RoadLane (type 3) from LANE_CONNECTOR baseline paths ---
    if need_lane_connector:
        for lc_obj in map_objects.get(SemanticMapLayer.LANE_CONNECTOR, []):
            try:
                baseline = lc_obj.baseline_path.discrete_path
            except Exception:
                continue
            all_road_points.extend(
                _polyline_to_road_points(baseline, type_index=3, width=0.0, height=0.0)
            )

    # --- CrossWalk (type 4) from crosswalk polygons ---
    if "CrossWalk" in enabled_set:
        for cw_obj in map_objects.get(SemanticMapLayer.CROSSWALK, []):
            try:
                polygon = cw_obj.polygon
            except Exception:
                continue
            all_road_points.extend(
                _polygon_to_road_points(polygon, type_index=4, width=0.0, height=0.0)
            )

    # --- StopSign (type 6) from STOP_LINE with STOP_SIGN type ---
    if "StopSign" in enabled_set:
        for sl_obj in map_objects.get(SemanticMapLayer.STOP_LINE, []):
            try:
                if sl_obj.stop_line_type != StopLineType.STOP_SIGN:
                    continue
                polygon = sl_obj.polygon
            except Exception:
                continue
            all_road_points.extend(
                _polygon_to_road_points(polygon, type_index=6, width=0.0, height=0.0)
            )

    if len(all_road_points) == 0:
        logger.warning("[RoadState] No road points extracted from map API.")
        return np.zeros((0, ROAD_DIM), dtype=np.float32)

    all_road_points = np.array(all_road_points, dtype=np.float64)  # (N, 7)

    # --- Diagnostic: per-type counts ---
    _type_names = ["none", "RoadLine", "RoadEdge", "RoadLane", "CrossWalk", "SpeedBump", "StopSign"]
    _type_counts = {_type_names[int(t)]: int(c) for t, c in
                    zip(*np.unique(all_road_points[:, 6], return_counts=True)) if int(t) < len(_type_names)}
    logger.info(f"[RoadState] Total road segments: {len(all_road_points)}, by type: {_type_counts}")

    # Transform to ego-local frame
    global_xy = all_road_points[:, :2]
    local_xy = (global_xy - np.array([ego_x, ego_y])) @ R.T
    local_orientation = _wrap_to_pi(all_road_points[:, 5] - ego_heading)

    all_road_points[:, 0] = local_xy[:, 0]
    all_road_points[:, 1] = local_xy[:, 1]
    all_road_points[:, 5] = local_orientation

    # Per-type cap: keep closest MAX_POINTS_PER_TYPE per type
    distances = np.sqrt(local_xy[:, 0] ** 2 + local_xy[:, 1] ** 2)
    selected_mask = np.zeros(len(all_road_points), dtype=bool)
    for t in np.unique(all_road_points[:, 6]).astype(int):
        type_mask = all_road_points[:, 6].astype(int) == t
        type_indices = np.where(type_mask)[0]
        type_dists = distances[type_indices]
        sorted_order = np.argsort(type_dists)
        keep = sorted_order[:MAX_POINTS_PER_TYPE]
        selected_mask[type_indices[keep]] = True
    all_road_points = all_road_points[selected_mask]
    logger.info(f"[RoadState] After per-type cap ({MAX_POINTS_PER_TYPE}): {len(all_road_points)} points")

    # Build (N, 13) output
    N = len(all_road_points)
    road_state = np.zeros((N, ROAD_DIM), dtype=np.float32)
    road_state[:, 0] = all_road_points[:, 0]  # x
    road_state[:, 1] = all_road_points[:, 1]  # y
    road_state[:, 2] = all_road_points[:, 2]  # segment_length
    road_state[:, 3] = all_road_points[:, 3]  # segment_width
    road_state[:, 4] = all_road_points[:, 4]  # segment_height
    road_state[:, 5] = all_road_points[:, 5]  # orientation
    # One-hot encoding for type (indices 6-12)
    type_indices = all_road_points[:, 6].astype(int)
    for i in range(N):
        if 0 <= type_indices[i] <= 6:
            road_state[i, 6 + type_indices[i]] = 1.0

    return road_state


def extract_waymo_whitebox_features(scene: Scene):
    """
    Extract Waymo-format whitebox features from a NAVSIM Scene.
    Returns (ego_state, partner_state, road_state) as separate arrays:
      ego_state:    (6,)
      partner_state: (63, 6)
      road_state:   (N, 13)  — variable length, all extracted road segments
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

    # ── 3. road_state (N x 13) ────────────────────────────────────────────
    road_state = _extract_road_state(scene, frame_idx)

    return ego_state, partner_state, road_state

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

    VIZ_DIR = Path(os.path.join("viz_output", "stage_one", token))
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

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
    Main entrypoint for stage-one (synthetic) whitebox visualization.
    :param cfg: omegaconf dictionary
    """

    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_all_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_loader_tokens_stage_one = scene_loader.tokens_stage_one
    tokens_to_evaluate_stage_one = sorted(set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens))

    token = tokens_to_evaluate_stage_one[0]
    scene = scene_loader.get_scene_from_token(token)
    
    navsim_self_visualization(scene, token)

    VIZ_DIR = Path(os.path.join("viz_output", "stage_one", token))
    save_path = VIZ_DIR / f"waymo_whitebox_obs.png"

    ego_state, partner_state, road_state = extract_waymo_whitebox_features(scene)
    visualize_single_obs_and_save(
        ego=ego_state, partners=partner_state, roads=road_state, save_path=str(save_path)
    )

if __name__ == "__main__":
    main()
