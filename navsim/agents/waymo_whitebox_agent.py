"""
WaymoWhiteboxAgent: extracts Waymo-format whitebox features from NAVSIM Scene,
passes them through a simple MLP, and outputs NAVSIM trajectory.

Input features (flattened):
  - ego_state:    1 x 6  = 6
  - partner_state: 63 x 6 = 378
  - road_state:   200 x 13 = 2600
  Total: 2984

Output: 8 poses x 3 (x, y, heading) in ego-local coordinates.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.common.enums import BoundingBoxIndex


# ── Constants ──────────────────────────────────────────────────────────────────
PARTNER_NUM = 63
PARTNER_DIM = 6
ROAD_OBS_COUNT = 200
ROAD_DIM = 13
EGO_DIM = 6
TOTAL_INPUT_DIM = EGO_DIM + PARTNER_NUM * PARTNER_DIM + ROAD_OBS_COUNT * ROAD_DIM  # 2984

# Target pkl file for saving whitebox features
TARGET_SCENE_TOKEN = "00a0f25bb297f4bb2"
_whitebox_saved = False  # Global flag to ensure save only once


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _save_whitebox_features_structured(
    ego_state: np.ndarray,
    partner_state: np.ndarray,
    road_state: np.ndarray,
    scene: Scene,
    save_dir: str = "/data/llh/navsim_workspace/exp/whitebox_features",
) -> None:
    """
    Save whitebox features as structured JSON for analysis.

    Args:
        ego_state: (6,) array [speed, vehicle_length, vehicle_width, rel_goal_x, rel_goal_y, is_collided]
        partner_state: (63, 6) array [speed, rel_pos_x, rel_pos_y, orientation, vehicle_length, vehicle_width]
        road_state: (200, 13) array [x, y, seg_len, seg_width, seg_height, orientation, type_onehot(7)]
        scene: NAVSIM Scene object
        save_dir: Directory to save the output JSON
    """
    global _whitebox_saved
    if _whitebox_saved:
        return

    # Check if this is the target scene
    scene_token = scene.scene_metadata.scene_token
    if not scene_token.startswith(TARGET_SCENE_TOKEN):
        return

    _whitebox_saved = True  # Mark as saved

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Build structured dictionary
    frame_idx = scene.scene_metadata.num_history_frames - 1
    ego_status = scene.frames[frame_idx].ego_status
    annotations = scene.frames[frame_idx].annotations

    # Ego pose info
    ego_pose_arr = ego_status.ego_pose
    ego_x = float(ego_pose_arr[0])
    ego_y = float(ego_pose_arr[1])
    ego_heading = float(ego_pose_arr[2])

    # Count non-zero partners
    partner_count = 0
    for i in range(PARTNER_NUM):
        if np.any(partner_state[i] != 0):
            partner_count += 1

    # Count non-zero road segments
    road_count = 0
    for i in range(ROAD_OBS_COUNT):
        if np.any(road_state[i] != 0):
            road_count += 1

    structured_data: Dict = {
        "metadata": {
            "scene_token": scene_token,
            "initial_token": scene.scene_metadata.initial_token,
            "log_name": scene.scene_metadata.log_name,
            "map_name": scene.scene_metadata.map_name,
            "num_history_frames": scene.scene_metadata.num_history_frames,
            "num_future_frames": scene.scene_metadata.num_future_frames,
            "frame_idx": frame_idx,
            "timestamp": datetime.now().isoformat(),
        },
        "ego_global": {
            "x": ego_x,
            "y": ego_y,
            "heading": ego_heading,
            "velocity": ego_status.ego_velocity.tolist(),
        },
        "ego_state": {
            "description": "[speed, vehicle_length, vehicle_width, rel_goal_x, rel_goal_y, is_collided]",
            "shape": list(ego_state.shape),
            "data": ego_state.tolist(),
            "field_names": ["speed", "vehicle_length", "vehicle_width", "rel_goal_x", "rel_goal_y", "is_collided"],
        },
        "partner_state": {
            "description": "[speed, rel_pos_x, rel_pos_y, orientation, vehicle_length, vehicle_width] per agent",
            "shape": list(partner_state.shape),
            "num_valid_partners": partner_count,
            "total_agents_in_frame": annotations.boxes.shape[0] if annotations is not None else 0,
            "data": partner_state.tolist(),
            "field_names": ["speed", "rel_pos_x", "rel_pos_y", "orientation", "vehicle_length", "vehicle_width"],
        },
        "road_state": {
            "description": "[x, y, seg_len, seg_width, seg_height, orientation, type_onehot(7)] per segment",
            "shape": list(road_state.shape),
            "num_valid_segments": road_count,
            "data": road_state.tolist(),
            "field_names": ["x", "y", "seg_len", "seg_width", "seg_height", "orientation",
                           "type_none", "type_RoadLine", "type_RoadEdge", "type_RoadLane",
                           "type_CrossWalk", "type_SpeedBump", "type_StopSign"],
        },
        "flattened_features": {
            "description": "Concatenated flat vector (ego + partner + road)",
            "total_dim": TOTAL_INPUT_DIM,
            "ego_dim": EGO_DIM,
            "partner_dim": PARTNER_NUM * PARTNER_DIM,
            "road_dim": ROAD_OBS_COUNT * ROAD_DIM,
        },
    }

    # Save to JSON
    save_path = os.path.join(save_dir, f"whitebox_features_{scene_token}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"WHITEBOX FEATURES SAVED to: {save_path}")
    print(f"  - ego_state: {ego_state.shape}")
    print(f"  - partner_state: {partner_state.shape} ({partner_count} valid partners)")
    print(f"  - road_state: {road_state.shape} ({road_count} valid segments)")
    print(f"{'='*80}\n")


def extract_waymo_whitebox_features(scene: Scene) -> np.ndarray:
    """
    Extract Waymo-format whitebox features from a NAVSIM Scene.
    Returns a flat float32 vector of length TOTAL_INPUT_DIM (2984).
    """
    frame_idx = scene.scene_metadata.num_history_frames - 1
    ego_status = scene.frames[frame_idx].ego_status
    annotations = scene.frames[frame_idx].annotations

    ego_pose_arr = ego_status.ego_pose  # (3,) [x, y, heading] global
    ego_x = float(ego_pose_arr[0])
    ego_y = float(ego_pose_arr[1])
    ego_heading = float(ego_pose_arr[2])

    # ── 1. ego-LocalEgoState (1 x 6) ──────────────────────────────────────
    speed = float(np.linalg.norm(ego_status.ego_velocity[:2]))

    # Fixed Pacifica parameters (navsim uses a single vehicle type)
    vehicle_length = 5.176
    vehicle_width = 2.297

    # rel_goal: GT trajectory endpoint in ego-local frame
    # Some synthetic scenes may fail get_future_trajectory due to ego_pose format;
    # fallback to speed-based extrapolation in that case.
    try:
        trajectory = scene.get_future_trajectory(
            num_trajectory_frames=scene.scene_metadata.num_future_frames
        )
        rel_goal_x = float(trajectory.poses[-1, 0])
        rel_goal_y = float(trajectory.poses[-1, 1])
    except Exception:
        # Fallback: extrapolate goal from current velocity over the time horizon
        dt = 0.5 * scene.scene_metadata.num_future_frames  # total seconds
        vel = ego_status.ego_velocity[:2]
        rel_goal_x = float(vel[0]) * dt
        rel_goal_y = float(vel[1]) * dt

    # is_collided: not available in open-loop, hardcode 0
    is_collided = 0.0

    ego_state = np.array(
        [speed, vehicle_length, vehicle_width, rel_goal_x, rel_goal_y, is_collided],
        dtype=np.float32,
    )

    # ── 2. partner_state (63 x 6) ─────────────────────────────────────────
    partner_state = _extract_partner_state(annotations, ego_heading)

    # ── 3. road_state (200 x 13) ──────────────────────────────────────────
    road_state = _extract_road_state(scene, ego_x, ego_y, ego_heading)

    # ── Save structured whitebox features for target scene (first stage only) ──
    _save_whitebox_features_structured(ego_state, partner_state, road_state, scene)

    # ── Flatten and concatenate ────────────────────────────────────────────
    features = np.concatenate([
        ego_state,                    # (6,)
        partner_state.flatten(),      # (378,)
        road_state.flatten(),         # (2600,)
    ]).astype(np.float32)

    assert features.shape == (TOTAL_INPUT_DIM,), f"Expected {TOTAL_INPUT_DIM}, got {features.shape}"
    return features


def _extract_partner_state(annotations, ego_heading: float) -> np.ndarray:
    """
    Extract partner_state (63 x 6).
    Fields: [speed, rel_pos_x, rel_pos_y, orientation, vehicle_length, vehicle_width]

    Key correction: orientation = wrap_to_pi(global_heading - ego_heading)
    annotations.boxes X/Y are already in ego-local frame.
    annotations.boxes HEADING is in global frame and must be converted.
    """
    N = annotations.boxes.shape[0]
    if N == 0:
        return np.zeros((PARTNER_NUM, PARTNER_DIM), dtype=np.float32)

    # Speed: norm of xy velocity
    speeds = np.linalg.norm(annotations.velocity_3d[:, :2], axis=1)  # (N,)

    # Position: already ego-local
    rel_pos_x = annotations.boxes[:, BoundingBoxIndex.X]  # (N,)
    rel_pos_y = annotations.boxes[:, BoundingBoxIndex.Y]  # (N,)

    # Orientation: global heading -> ego-relative heading
    global_heading = annotations.boxes[:, BoundingBoxIndex.HEADING]  # (N,)
    orientation = _wrap_to_pi(global_heading - ego_heading)  # (N,)

    # Dimensions
    vehicle_length = annotations.boxes[:, BoundingBoxIndex.LENGTH]  # (N,)
    vehicle_width = annotations.boxes[:, BoundingBoxIndex.WIDTH]    # (N,)

    # Stack: (N, 6)
    all_partners = np.stack(
        [speeds, rel_pos_x, rel_pos_y, orientation, vehicle_length, vehicle_width],
        axis=1,
    ).astype(np.float32)

    # Sort by distance to ego (ascending), take closest 63
    distances = np.sqrt(rel_pos_x ** 2 + rel_pos_y ** 2)
    sorted_indices = np.argsort(distances)
    all_partners = all_partners[sorted_indices]

    # Pad or truncate to PARTNER_NUM
    result = np.zeros((PARTNER_NUM, PARTNER_DIM), dtype=np.float32)
    count = min(N, PARTNER_NUM)
    result[:count] = all_partners[:count]
    return result


def _extract_road_state(
    scene: Scene, ego_x: float, ego_y: float, ego_heading: float
) -> np.ndarray:
    """
    Extract LocalRoadState (200 x 13).
    Each row = one polyline segment (pair of adjacent control points).

    Fields: [x, y, seg_len, seg_width, seg_height, orientation,
             type.none, type.RoadLine, type.RoadEdge, type.RoadLane,
             type.CrossWalk, type.SpeedBump, type.StopSign]

    Key corrections vs probe code:
    - Expand per-segment (not per-lane)
    - seg_width/seg_height: fill 0 (not reliably extractable)
    - RoadLine/RoadEdge/SpeedBump: fill 0 (not available in nuPlan maps)
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)

    # Layer -> one-hot index mapping
    # 0=none, 1=RoadLine, 2=RoadEdge, 3=RoadLane, 4=CrossWalk, 5=SpeedBump, 6=StopSign
    layer_to_onehot_idx = {
        SemanticMapLayer.LANE: 3,            # RoadLane
        SemanticMapLayer.LANE_CONNECTOR: 3,  # RoadLane
        SemanticMapLayer.CROSSWALK: 4,       # CrossWalk
        SemanticMapLayer.STOP_LINE: 6,       # StopSign (approximate)
    }

    query_point = Point2D(ego_x, ego_y)
    layers = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.STOP_LINE,
    ]

    try:
        map_objects = scene.map_api.get_proximal_map_objects(
            point=query_point, radius=50.0, layers=layers
        )
    except Exception:
        return np.zeros((ROAD_OBS_COUNT, ROAD_DIM), dtype=np.float32)

    rows = []  # list of (distance_to_ego, row_13d)

    for layer, objs in map_objects.items():
        onehot_idx = layer_to_onehot_idx.get(layer, 0)

        for obj in objs:
            coords = _get_coords_from_map_object(obj)
            if coords is None or len(coords) < 2:
                continue

            # Expand per-segment: each pair of adjacent points -> one row
            for i in range(len(coords) - 1):
                p0 = coords[i]
                p1 = coords[i + 1]

                # Midpoint in global frame
                mid_x = (p0[0] + p1[0]) / 2.0
                mid_y = (p0[1] + p1[1]) / 2.0

                # Transform to ego-local
                dx = mid_x - ego_x
                dy = mid_y - ego_y
                local_x = dx * cos_h - dy * sin_h
                local_y = dx * sin_h + dy * cos_h

                # Segment half-length
                seg_len = float(np.sqrt(
                    (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
                )) / 2.0

                # Width and height: not reliably extractable
                seg_width = 0.0
                seg_height = 0.0

                # Orientation relative to ego heading
                seg_orient = float(np.arctan2(
                    p1[1] - p0[1], p1[0] - p0[0]
                )) - ego_heading
                seg_orient = float(_wrap_to_pi(np.array(seg_orient)))

                # One-hot type (7 dims)
                type_onehot = [0.0] * 7
                type_onehot[onehot_idx] = 1.0

                row = [local_x, local_y, seg_len, seg_width,
                       seg_height, seg_orient] + type_onehot
                dist_to_ego = float(np.sqrt(local_x ** 2 + local_y ** 2))
                rows.append((dist_to_ego, row))

    # Sort by distance, take closest ROAD_OBS_COUNT
    rows.sort(key=lambda x: x[0])
    result = np.zeros((ROAD_OBS_COUNT, ROAD_DIM), dtype=np.float32)
    count = min(len(rows), ROAD_OBS_COUNT)
    for i in range(count):
        result[i] = rows[i][1]

    return result


def _get_coords_from_map_object(obj) -> np.ndarray:
    """Extract xy coordinates from a map object."""
    # Try baseline_path first (Lane, LaneConnector)
    try:
        ls = obj.baseline_path.linestring
        return np.array(ls.coords)[:, :2]  # (M, 2)
    except Exception:
        pass

    # Fallback: polygon exterior (CrossWalk, StopLine, RoadBlock)
    try:
        poly = obj.polygon
        return np.array(poly.exterior.coords)[:, :2]  # (M, 2)
    except Exception:
        pass

    return None


# ── Neural Network ─────────────────────────────────────────────────────────────

class WaymoWhiteboxMLP(nn.Module):
    """Simple MLP: Waymo whitebox features -> NAVSIM trajectory."""

    def __init__(
        self,
        input_dim: int = TOTAL_INPUT_DIM,
        hidden_dim: int = 256,
        num_poses: int = 8,
    ):
        super().__init__()
        self.num_poses = num_poses
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_poses * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, input_dim)
        :return: (B, num_poses, 3)
        """
        out = self.net(x)
        return out.reshape(-1, self.num_poses, 3)


# ── Agent ──────────────────────────────────────────────────────────────────────

class WaymoWhiteboxAgent(AbstractAgent):
    """
    Agent that extracts Waymo whitebox features from NAVSIM Scene
    and predicts trajectory via MLP.
    """

    requires_scene = True

    def __init__(
        self,
        hidden_dim: int = 256,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
    ):
        super().__init__(trajectory_sampling, requires_scene=True)
        self._checkpoint_path = checkpoint_path
        self._hidden_dim = hidden_dim
        self._mlp = WaymoWhiteboxMLP(
            input_dim=TOTAL_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_poses=self._trajectory_sampling.num_poses,
        )

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._checkpoint_path is not None:
            if torch.cuda.is_available():
                state_dict = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict = torch.load(
                    self._checkpoint_path, map_location=torch.device("cpu")
                )["state_dict"]
            self.load_state_dict(
                {k.replace("agent.", ""): v for k, v in state_dict.items()}
            )

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(
        self, agent_input: AgentInput, scene: Scene  # noqa: ARG002
    ) -> Trajectory:
        """
        Extract Waymo whitebox features from Scene, run MLP, return Trajectory.
        """
        self.eval()

        # Extract features
        features_np = extract_waymo_whitebox_features(scene)
        features_tensor = torch.tensor(
            features_np, dtype=torch.float32
        ).unsqueeze(0)  # (1, 2984)

        # Forward pass
        with torch.no_grad():
            poses = self._mlp(features_tensor).squeeze(0).numpy()  # (num_poses, 3)

        return Trajectory(poses.astype(np.float32), self._trajectory_sampling)