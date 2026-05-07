#!/usr/bin/env python
"""Export one NAVSIM scene token to a ScenarioMax/GPUDrive-style JSON file.

This is intentionally scoped to token-level alignment work.  It reads a NAVSIM
Scene, uses the current frame as the origin, extracts only that current frame's
dynamic agents and map features, then writes the same top-level JSON schema
produced by ScenarioMax's nuPlan -> GPUDrive exporter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_TOKEN = "00016f8b45c25a1d"
DEFAULT_LOG_NAME = "2021.09.29.14.44.26_veh-28_00337_00504"
DEFAULT_DATASET_NAME = "nuPlan"
DEFAULT_DATASET_VERSION = "navsim_token_eval"
DEFAULT_SCENARIO_TYPE_PREFIX = "navsim_scene"
DEFAULT_MAP_RADIUS_METERS = 250
ERR_VAL = -1e4


VEHICLE = "VEHICLE"
PEDESTRIAN = "PEDESTRIAN"
CYCLIST = "CYCLIST"
OTHER = "OTHER"
LANE_SURFACE_STREET = "LANE_SURFACE_STREET"
CROSSWALK = "CROSSWALK"
ROAD_EDGE_BOUNDARY = "ROAD_EDGE_BOUNDARY"
ROAD_LINE_UNKNOWN = "ROAD_LINE_UNKNOWN"
TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
TRAFFIC_LIGHT_RED = "TRAFFIC_LIGHT_RED"
TRAFFIC_LIGHT_GREEN = "TRAFFIC_LIGHT_GREEN"
TRAFFIC_LIGHT_UNKNOWN = "TRAFFIC_LIGHT_UNKNOWN"


ROAD_LINE_TYPE_BY_BOUNDARY_FID = {
    0: "ROAD_LINE_BROKEN_SINGLE_WHITE",
    1: "ROAD_LINE_SOLID_SINGLE_WHITE",
    2: "ROAD_LINE_SOLID_SINGLE_WHITE",
    3: ROAD_LINE_UNKNOWN,
    4: "ROAD_LINE_SOLID_SINGLE_YELLOW",
    5: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
    6: "ROAD_LINE_SOLID_DOUBLE_WHITE",
    7: "ROAD_LINE_SOLID_DOUBLE_YELLOW",
}


TYPE_MAPPING = {
    "LANE_FREEWAY": 1,
    "LANE_CENTER_FREEWAY": 1,
    "LANE_SURFACE_STREET": 2,
    "LANE_SURFACE_UNSTRUCTURE": 2,
    "LANE_BIKE_LANE": 3,
    "ROAD_LINE_BROKEN_SINGLE_WHITE": 6,
    "ROAD_LINE_SOLID_SINGLE_WHITE": 7,
    "ROAD_LINE_SOLID_DOUBLE_WHITE": 8,
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": 9,
    "ROAD_LINE_BROKEN_DOUBLE_YELLOW": 10,
    "ROAD_LINE_SOLID_SINGLE_YELLOW": 11,
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": 12,
    "ROAD_LINE_PASSING_DOUBLE_YELLOW": 13,
    "ROAD_EDGE_BOUNDARY": 15,
    "ROAD_EDGE_MEDIAN": 16,
    "STOP_SIGN": 17,
    "CROSSWALK": 18,
    "SPEED_BUMP": 19,
}


TYPE_TO_MAP_FEATURE_NAME = {
    "ROAD_EDGE": "road_edge",
    "ROAD_LINE": "road_line",
    "LANE": "lane",
    "STOP_SIGN": "stop_sign",
    "CROSSWALK": "crosswalk",
    "SPEED_BUMP": "speed_bump",
    "DRIVEWAY": "driveway",
}


NAVSIM_AGENT_TYPE_TO_UNIFIED = {
    "vehicle": VEHICLE,
    "pedestrian": PEDESTRIAN,
    "bicycle": CYCLIST,
}


UNIFIED_TYPE_TO_GPUDRIVE = {
    VEHICLE: "vehicle",
    PEDESTRIAN: "pedestrian",
    CYCLIST: "cyclist",
    OTHER: "other",
}


TRAFFIC_LIGHT_STATE_TO_GPUDRIVE = {
    TRAFFIC_LIGHT_RED: "stop",
    TRAFFIC_LIGHT_GREEN: "go",
    TRAFFIC_LIGHT_UNKNOWN: "unknown",
}


def _rotate_xy(vector: Iterable[float], angle: float) -> np.ndarray:
    x, y = vector
    sin_angle, cos_angle = math.sin(angle), math.cos(angle)
    return np.asarray(
        [
            x * cos_angle - y * sin_angle,
            x * sin_angle + y * cos_angle,
        ],
        dtype=np.float32,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return _repo_root().parent


def _default_openscene_root() -> Path:
    return Path(os.environ.get("OPENSCENE_DATA_ROOT", _workspace_root() / "dataset")).expanduser()


def _center_vector(vector: Any, center: Iterable[float] = (0.0, 0.0)) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array - np.asarray(center, dtype=np.float64)


def _wrap_yaw(yaw: float | np.ndarray) -> float | np.ndarray:
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def _ensure_scalar(value: Any) -> float:
    if isinstance(value, np.ndarray):
        return float(value.item() if value.size == 1 else value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return value.item()
    return float(value)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _mph_to_kmh(speed_mph: float) -> float:
    return speed_mph * 1.609344


def _geometry_xy_points(geometry: Any) -> np.ndarray | None:
    from shapely.geometry import LineString, MultiLineString
    from shapely.geometry.polygon import LinearRing

    if isinstance(geometry, (LineString, LinearRing)):
        x, y = geometry.xy
        return np.asarray(list(zip(x, y)), dtype=np.float64)

    if isinstance(geometry, MultiLineString):
        lines = list(geometry.geoms)
        if not lines:
            return None
        longest = max(lines, key=lambda line: len(line.coords))
        x, y = longest.xy
        return np.asarray(list(zip(x, y)), dtype=np.float64)

    if hasattr(geometry, "coords"):
        return np.asarray(list(geometry.coords), dtype=np.float64)

    return None


def _boundary_points(boundary: Any, center: np.ndarray) -> np.ndarray:
    points = [(pose.x, pose.y) for pose in boundary.discrete_path]
    return _center_vector(points, center)


def _centerline_points(map_obj: Any, center: np.ndarray) -> np.ndarray:
    points = [(pose.x, pose.y) for pose in map_obj.baseline_path.discrete_path]
    return _center_vector(points, center)


def _road_line_type(boundary_type_fid: int) -> str:
    return ROAD_LINE_TYPE_BY_BOUNDARY_FID.get(boundary_type_fid, ROAD_LINE_UNKNOWN)


def _iter_boundary_lines(boundary_geometry: Any) -> list[Any]:
    from shapely.geometry import GeometryCollection, LineString, MultiLineString
    from shapely.geometry.polygon import LinearRing

    if isinstance(boundary_geometry, (LineString, LinearRing)):
        return [boundary_geometry]
    if isinstance(boundary_geometry, MultiLineString):
        return list(boundary_geometry.geoms)
    if isinstance(boundary_geometry, GeometryCollection):
        lines: list[Any] = []
        for geom in boundary_geometry.geoms:
            lines.extend(_iter_boundary_lines(geom))
        return lines
    return []


def extract_static_map_elements(map_api: Any, center: np.ndarray, radius: int) -> dict[str, dict[str, Any]]:
    """Mirror ScenarioMax's nuPlan static-map extraction using NAVSIM's map API."""
    import geopandas as gpd
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
    from shapely.ops import unary_union

    static_map_elements: dict[str, dict[str, Any]] = {}
    layer_names = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION,
    ]

    map_objects = map_api.get_proximal_map_objects(Point2D(float(center[0]), float(center[1])), radius, layer_names)

    try:
        boundaries = map_api._get_vector_map_layer(SemanticMapLayer.BOUNDARIES)
    except Exception:  # noqa: BLE001
        boundaries = None

    if SemanticMapLayer.STOP_LINE in map_objects:
        map_objects[SemanticMapLayer.STOP_LINE] = [
            stop_line
            for stop_line in map_objects[SemanticMapLayer.STOP_LINE]
            if stop_line.stop_line_type != StopLineType.TURN_STOP
        ]

    block_polygons = []
    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for block in map_objects.get(layer, []):
            edges = (
                sorted(block.interior_edges, key=lambda lane: lane.index)
                if layer == SemanticMapLayer.ROADBLOCK
                else block.interior_edges
            )

            for index, lane_data in enumerate(edges):
                if not hasattr(lane_data, "baseline_path"):
                    continue

                polygon_points = _geometry_xy_points(lane_data.polygon.boundary)
                if polygon_points is None or len(polygon_points) == 0:
                    continue

                lane_polyline = _centerline_points(lane_data, center)
                speed_limit_mps = lane_data.speed_limit_mps
                speed_limit_mph = speed_limit_mps * 2.23694 if speed_limit_mps else None
                speed_limit_kmh = _mph_to_kmh(speed_limit_mph) if speed_limit_mph else None

                static_map_elements[str(lane_data.id)] = {
                    "type": LANE_SURFACE_STREET,
                    "polyline": lane_polyline,
                    "speed_limit_mph": speed_limit_mph,
                    "speed_limit_kmh": speed_limit_kmh,
                    "entry_lanes": [str(edge.id) for edge in lane_data.incoming_edges],
                    "exit_lanes": [str(edge.id) for edge in lane_data.outgoing_edges],
                    "left_neighbor": (
                        [str(edge.id) for edge in block.interior_edges[:index]]
                        if layer == SemanticMapLayer.ROADBLOCK
                        else []
                    ),
                    "right_neighbor": (
                        [str(edge.id) for edge in block.interior_edges[index + 1 :]]
                        if layer == SemanticMapLayer.ROADBLOCK
                        else []
                    ),
                    "polygon": _center_vector(polygon_points, center),
                }

                if layer == SemanticMapLayer.ROADBLOCK and boundaries is not None:
                    adjacent_edges = lane_data.adjacent_edges
                    if adjacent_edges[0] and adjacent_edges[1]:
                        for boundary in [lane_data.left_boundary, lane_data.right_boundary]:
                            boundary_id = str(boundary.id)
                            if boundary_id in static_map_elements:
                                continue
                            try:
                                boundary_type_fid = int(boundaries.loc[[boundary_id]]["boundary_type_fid"].iloc[0])
                                line_type = _road_line_type(boundary_type_fid)
                            except Exception:  # noqa: BLE001
                                line_type = ROAD_LINE_UNKNOWN
                            if line_type != ROAD_LINE_UNKNOWN:
                                static_map_elements[boundary_id] = {
                                    "type": line_type,
                                    "polyline": _boundary_points(boundary, center),
                                }

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    for crosswalk in map_objects.get(SemanticMapLayer.CROSSWALK, []):
        polygon_points = _geometry_xy_points(crosswalk.polygon.exterior)
        if polygon_points is None or len(polygon_points) == 0:
            continue
        static_map_elements[str(crosswalk.id)] = {
            "type": CROSSWALK,
            "polygon": _center_vector(polygon_points, center),
        }

    intersection_polygons = [intersection.polygon for intersection in map_objects.get(SemanticMapLayer.INTERSECTION, [])]
    all_polygons = intersection_polygons + block_polygons
    if all_polygons:
        unified_boundary = gpd.GeoSeries(unary_union(all_polygons)).boundary.iloc[0]
        for idx, boundary in enumerate(_iter_boundary_lines(unified_boundary)):
            boundary_points = np.asarray(list(zip(boundary.coords.xy[0], boundary.coords.xy[1])), dtype=np.float64)
            static_map_elements[f"boundary_{idx}"] = {
                "type": ROAD_EDGE_BOUNDARY,
                "polyline": _center_vector(boundary_points, center)[::-1],
            }

    return static_map_elements


def _map_type_to_mapfeature(feature_type: str) -> str:
    if feature_type in TYPE_TO_MAP_FEATURE_NAME:
        return TYPE_TO_MAP_FEATURE_NAME[feature_type]
    for key, value in TYPE_TO_MAP_FEATURE_NAME.items():
        if key in feature_type:
            return value
    return feature_type.lower()


def convert_map_features(static_map_elements: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[Any]]]:
    roads = []
    edge_segments = []
    edge_points = []
    index = 0

    for map_feature_id, feature in static_map_elements.items():
        feature_type = feature["type"]
        if feature_type in {"ROAD_EDGE_SIDEWALK", "DRIVEWAY"}:
            continue

        new_feature = {
            "geometry": [],
            "type": _map_type_to_mapfeature(feature_type),
            "map_element_id": TYPE_MAPPING.get(feature_type, 0),
            "id": int(map_feature_id) if str(map_feature_id).isdigit() else index,
        }
        geometry_key = next((key for key in ["polyline", "polygon", "position"] if key in feature), None)
        if geometry_key is None:
            continue

        original_geometry = np.asarray(feature[geometry_key], dtype=np.float64)
        if original_geometry.ndim == 1:
            original_geometry = np.expand_dims(original_geometry, 0)

        if original_geometry.shape[-1] == 3:
            geometry = [{"x": point[0], "y": point[1], "z": point[2]} for point in original_geometry]
        else:
            geometry = [{"x": point[0], "y": point[1], "z": 0.0} for point in original_geometry]

        new_feature["geometry"] = geometry

        if "ROAD_EDGE" in feature_type:
            edge_vertices = [[point["x"], point["y"], point["z"]] for point in geometry]
            edge_points.extend(edge_vertices)
            edge_segments.extend([[edge_vertices[i], edge_vertices[i + 1]] for i in range(len(edge_vertices) - 1)])

        roads.append(new_feature)
        index += 1

    if edge_points:
        edge_points_array = np.asarray(edge_points, dtype=np.float32)
        xy_points = edge_points_array[:, :2]
        tolerance = 0.2
        chunk_size = 1000
        for i in range(0, len(xy_points), chunk_size):
            chunk = xy_points[i : i + chunk_size]
            dists = np.linalg.norm(chunk[:, np.newaxis] - xy_points, axis=2)
            potential_pairs = np.where((dists < tolerance) & (dists > 0))
            for p1, p2 in zip(*potential_pairs):
                p1_idx = i + p1
                if abs(edge_points_array[p1_idx, 2] - edge_points_array[p2, 2]) > tolerance:
                    raise RuntimeError("Map appears to contain 3D overlapping road-edge geometry.")

    return roads, edge_segments


def _traffic_light_position(map_api: Any, lane_id: str, center: np.ndarray) -> np.ndarray | None:
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    lane = map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
    if lane is None:
        return None
    path = lane.baseline_path.discrete_path
    if not path:
        return None
    return np.asarray([path[0].x - center[0], path[0].y - center[1], 0.0], dtype=np.float32)


def extract_dynamic_map_elements(
    frames: list[Any],
    map_api: Any,
    center: np.ndarray,
    target_times: np.ndarray,
) -> dict[str, dict[str, Any]]:
    source_times = np.asarray([(frame.timestamp - frames[0].timestamp) * 1e-6 for frame in frames], dtype=np.float32)
    all_lane_ids = sorted({str(lane_id) for frame in frames for lane_id, _ in frame.traffic_lights})
    dynamic_map_elements: dict[str, dict[str, Any]] = {}

    for lane_id in all_lane_ids:
        position = _traffic_light_position(map_api, lane_id, center)
        if position is None:
            continue

        source_states = []
        for frame in frames:
            lane_state = TRAFFIC_LIGHT_UNKNOWN
            for frame_lane_id, is_red in frame.traffic_lights:
                if str(frame_lane_id) == lane_id:
                    lane_state = TRAFFIC_LIGHT_RED if is_red else TRAFFIC_LIGHT_GREEN
                    break
            source_states.append(lane_state)

        target_states = []
        for target_time in target_times:
            if target_time < source_times[0] or target_time > source_times[-1]:
                target_states.append(TRAFFIC_LIGHT_UNKNOWN)
                continue
            source_index = int(np.searchsorted(source_times, target_time, side="right") - 1)
            source_index = min(max(source_index, 0), len(source_states) - 1)
            target_states.append(source_states[source_index])

        dynamic_map_elements[lane_id] = {
            "type": TRAFFIC_LIGHT,
            "position": position,
            "states": target_states,
            "lane": lane_id,
        }

    return dynamic_map_elements


def convert_traffic_lights(dynamic_map_elements: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[Any]]]:
    tl_dict: dict[str, dict[str, list[Any]]] = {}
    for lane_id, tl_state in dynamic_map_elements.items():
        position = tl_state["position"]
        x, y = position[:2]
        z = position[2] if len(position) > 2 else 0.0
        tl_dict[lane_id] = {"state": [], "x": [], "y": [], "z": [], "time_index": [], "lane_id": []}
        for time_index, state in enumerate(tl_state["states"]):
            tl_dict[lane_id]["state"].append(TRAFFIC_LIGHT_STATE_TO_GPUDRIVE.get(state, "unknown"))
            tl_dict[lane_id]["x"].append(x)
            tl_dict[lane_id]["y"].append(y)
            tl_dict[lane_id]["z"].append(z)
            tl_dict[lane_id]["time_index"].append(time_index)
            tl_dict[lane_id]["lane_id"].append(lane_id)
    return tl_dict


def _empty_agent_state(num_steps: int) -> dict[str, np.ndarray]:
    return {
        "position": np.zeros((num_steps, 3), dtype=np.float32),
        "heading": np.zeros((num_steps,), dtype=np.float32),
        "velocity": np.zeros((num_steps, 2), dtype=np.float32),
        "valid": np.zeros((num_steps,), dtype=np.float32),
        "length": np.zeros((num_steps, 1), dtype=np.float32),
        "width": np.zeros((num_steps, 1), dtype=np.float32),
        "height": np.zeros((num_steps, 1), dtype=np.float32),
    }


def _extract_source_dynamic_agents(frames: list[Any], center: np.ndarray) -> dict[str, dict[str, Any]]:
    from navsim.planning.scenario_builder.navsim_scenario_utils import (
        annotations_to_detection_tracks,
        ego_status_to_ego_state,
    )
    from nuplan.common.actor_state.agent import Agent
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.static_object import StaticObject
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    vehicle_parameters = get_pacifica_parameters()
    ego_states = [
        ego_status_to_ego_state(frame.ego_status, vehicle_parameters, TimePoint(int(frame.timestamp)))
        for frame in frames
    ]

    dynamic_agents: dict[str, dict[str, Any]] = {
        "ego": {
            "type": VEHICLE,
            "states": _empty_agent_state(len(frames)),
        },
    }

    for frame_idx, (frame, ego_state) in enumerate(zip(frames, ego_states)):
        ego_track = dynamic_agents["ego"]["states"]
        ego_track["position"][frame_idx] = [
            ego_state.waypoint.x - center[0],
            ego_state.waypoint.y - center[1],
            0.0,
        ]
        ego_track["heading"][frame_idx] = ego_state.waypoint.heading
        ego_track["valid"][frame_idx] = 1.0
        ego_track["length"][frame_idx] = ego_state.agent.box.length
        ego_track["width"][frame_idx] = ego_state.agent.box.width
        ego_track["height"][frame_idx] = ego_state.agent.box.height
        ego_track["velocity"][frame_idx] = _rotate_xy(frame.ego_status.ego_velocity, ego_state.waypoint.heading)

        detections = annotations_to_detection_tracks(frame.annotations, ego_state).tracked_objects.tracked_objects
        for tracked_object in detections:
            if not isinstance(tracked_object, (Agent, StaticObject)):
                continue

            navsim_name = str(tracked_object.tracked_object_type.name).lower()
            unified_type = NAVSIM_AGENT_TYPE_TO_UNIFIED.get(navsim_name)
            if unified_type is None:
                continue

            object_id = str(tracked_object.track_token)
            if object_id not in dynamic_agents:
                dynamic_agents[object_id] = {
                    "type": unified_type,
                    "states": _empty_agent_state(len(frames)),
                }

            track = dynamic_agents[object_id]["states"]
            track["position"][frame_idx] = [
                tracked_object.center.x - center[0],
                tracked_object.center.y - center[1],
                0.0,
            ]
            track["heading"][frame_idx] = tracked_object.center.heading
            track["valid"][frame_idx] = 1.0
            track["length"][frame_idx] = tracked_object.box.length
            track["width"][frame_idx] = tracked_object.box.width
            track["height"][frame_idx] = tracked_object.box.height
            if hasattr(tracked_object, "velocity"):
                track["velocity"][frame_idx] = [tracked_object.velocity.x, tracked_object.velocity.y]

    return dynamic_agents


def _scene_uses_current_position_goal(scene: Any, goal_source: str) -> bool:
    if goal_source == "current":
        return True
    if goal_source == "future":
        return False
    return bool(
        scene.scene_metadata.corresponding_original_scene
        or scene.scene_metadata.corresponding_original_initial_token
    )


def extract_goal_positions(
    scene: Any,
    current_frame_idx: int,
    center: np.ndarray,
    goal_source: str,
) -> dict[str, np.ndarray]:
    from navsim.planning.scenario_builder.navsim_scenario_utils import (
        annotations_to_detection_tracks,
        ego_status_to_ego_state,
    )
    from nuplan.common.actor_state.agent import Agent
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.static_object import StaticObject
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    end_frame_idx = current_frame_idx if _scene_uses_current_position_goal(scene, goal_source) else len(scene.frames) - 1
    end_frame_idx = max(current_frame_idx, min(end_frame_idx, len(scene.frames) - 1))
    vehicle_parameters = get_pacifica_parameters()
    goal_positions: dict[str, np.ndarray] = {}

    for frame_idx in range(current_frame_idx, end_frame_idx + 1):
        frame = scene.frames[frame_idx]
        ego_state = ego_status_to_ego_state(
            frame.ego_status,
            vehicle_parameters,
            TimePoint(int(frame.timestamp)),
        )
        goal_positions["ego"] = np.asarray(
            [
                ego_state.waypoint.x - center[0],
                ego_state.waypoint.y - center[1],
                0.0,
            ],
            dtype=np.float32,
        )

        detections = annotations_to_detection_tracks(frame.annotations, ego_state).tracked_objects.tracked_objects
        for tracked_object in detections:
            if not isinstance(tracked_object, (Agent, StaticObject)):
                continue
            navsim_name = str(tracked_object.tracked_object_type.name).lower()
            if NAVSIM_AGENT_TYPE_TO_UNIFIED.get(navsim_name) is None:
                continue
            goal_positions[str(tracked_object.track_token)] = np.asarray(
                [
                    tracked_object.center.x - center[0],
                    tracked_object.center.y - center[1],
                    0.0,
                ],
                dtype=np.float32,
            )

    return goal_positions


def _object_distance_traveled(positions: np.ndarray, valids: np.ndarray) -> float:
    valid_positions = positions[valids.astype(bool)]
    if len(valid_positions) < 2:
        return 0.0
    diffs = valid_positions[1:] - valid_positions[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def _extract_gpudrive_object(index: int, object_id: str, agent: dict[str, Any]) -> dict[str, Any]:
    state = agent["states"]
    valids = state["valid"].astype(bool)
    positions = state["position"]
    headings = state["heading"]
    velocities = state["velocity"]

    final_valid_index = 0
    for valid_index, is_valid in enumerate(valids):
        if is_valid:
            final_valid_index = valid_index

    position = [
        {"x": point[0], "y": point[1], "z": point[2]} if valids[i] else {"x": ERR_VAL, "y": ERR_VAL, "z": ERR_VAL}
        for i, point in enumerate(positions)
    ]
    heading = [_wrap_yaw(float(value)) if valids[i] else ERR_VAL for i, value in enumerate(headings)]
    velocity = [
        {"x": point[0], "y": point[1]} if valids[i] else {"x": ERR_VAL, "y": ERR_VAL}
        for i, point in enumerate(velocities)
    ]
    goal_position = np.asarray(agent.get("goal_position", positions[final_valid_index]), dtype=np.float32)

    return {
        "id": int(object_id) if str(object_id).isdigit() else index,
        "type": UNIFIED_TYPE_TO_GPUDRIVE.get(agent["type"], str(agent["type"]).lower()),
        "position": position,
        "width": _ensure_scalar(state["width"][final_valid_index]),
        "length": _ensure_scalar(state["length"][final_valid_index]),
        "height": _ensure_scalar(state["height"][final_valid_index]),
        "heading": heading,
        "velocity": velocity,
        "valid": valids.tolist(),
        "goalPosition": {
            "x": _ensure_scalar(goal_position[0]),
            "y": _ensure_scalar(goal_position[1]),
            "z": _ensure_scalar(goal_position[2]),
        },
        "is_sdc": object_id == "ego",
        "mark_as_expert": False,
        "total_distance_traveled": _object_distance_traveled(positions, valids),
    }


def convert_dynamic_agents(dynamic_agents: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float]]:
    objects = []
    distances = []

    object_ids = [object_id for object_id in dynamic_agents.keys() if object_id != "ego"]
    object_ids = sorted(object_ids)
    if "ego" in dynamic_agents:
        object_ids.insert(0, "ego")

    for index, object_id in enumerate(object_ids):
        agent = dynamic_agents[object_id]
        obj = _extract_gpudrive_object(index, object_id, agent)
        if obj["type"] in {"vehicle", "cyclist"}:
            if any(obj["valid"]):
                distances.append(float(obj["total_distance_traveled"]))
                objects.append(obj)
        elif obj["type"] == "pedestrian":
            objects.append(obj)

    return objects, distances


def _current_xy(obj: dict[str, Any]) -> np.ndarray | None:
    if obj["valid"] and obj["valid"][0]:
        position = obj["position"][0]
        return np.asarray([position["x"], position["y"]], dtype=np.float32)
    return None


def align_object_order_to_reference(objects: list[dict[str, Any]], reference_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Greedily align object order to a reference JSON by type and current position."""
    remaining = set(range(len(objects)))
    aligned = []
    matched_reference_ids = []

    for ref_obj in reference_objects:
        ref_xy = _current_xy(ref_obj)
        if ref_xy is None:
            continue
        best_index = None
        best_distance = math.inf
        for candidate_index in remaining:
            candidate = objects[candidate_index]
            if candidate["type"] != ref_obj["type"] or bool(candidate["is_sdc"]) != bool(ref_obj["is_sdc"]):
                continue
            candidate_xy = _current_xy(candidate)
            if candidate_xy is None:
                continue
            distance = float(np.linalg.norm(candidate_xy - ref_xy))
            if distance < best_distance:
                best_distance = distance
                best_index = candidate_index
        if best_index is not None:
            aligned.append(objects[best_index])
            matched_reference_ids.append(ref_obj.get("id"))
            remaining.remove(best_index)

    for obj, reference_id in zip(aligned, matched_reference_ids):
        if reference_id is not None:
            obj["id"] = reference_id

    used_ids = {obj["id"] for obj in aligned}
    for index in sorted(remaining):
        obj = objects[index]
        while obj["id"] in used_ids:
            obj["id"] += 1
        used_ids.add(obj["id"])
        aligned.append(obj)
    return aligned


def build_scenario_json(
    scene: Any,
    *,
    map_radius: int,
    goal_source: str,
    dataset_name: str,
    dataset_version: str,
    scenario_type_prefix: str,
    reference_json: dict[str, Any] | None,
    align_reference_order: bool,
) -> dict[str, Any]:
    from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    current_frame_idx = scene.scene_metadata.num_history_frames - 1
    if current_frame_idx >= len(scene.frames):
        raise ValueError("The NAVSIM scene has no frames from the current frame onward.")
    current_frame = scene.frames[current_frame_idx]
    frames = [current_frame]
    vehicle_parameters = get_pacifica_parameters()
    current_ego_state = ego_status_to_ego_state(
        current_frame.ego_status,
        vehicle_parameters,
        TimePoint(int(current_frame.timestamp)),
    )
    center = np.asarray([current_ego_state.waypoint.x, current_ego_state.waypoint.y], dtype=np.float64)
    target_times = np.asarray([0.0], dtype=np.float32)

    dynamic_agents = _extract_source_dynamic_agents(frames, center)
    goal_positions = extract_goal_positions(scene, current_frame_idx, center, goal_source)
    for object_id, agent in dynamic_agents.items():
        if object_id in goal_positions:
            agent["goal_position"] = goal_positions[object_id]

    static_map_elements = extract_static_map_elements(scene.map_api, center, map_radius)
    roads, _ = convert_map_features(static_map_elements)
    tl_states = convert_traffic_lights(extract_dynamic_map_elements(frames, scene.map_api, center, target_times))
    objects, distances = convert_dynamic_agents(dynamic_agents)

    if reference_json is not None and align_reference_order:
        objects = align_object_order_to_reference(objects, reference_json.get("objects", []))

    sdc_indices = [index for index, obj in enumerate(objects) if obj["is_sdc"]]
    if not sdc_indices:
        raise ValueError("No ego/SDC object was exported.")

    scenario_id = scene.scene_metadata.initial_token
    export_file_name = f"{dataset_name}_{dataset_version}_{scenario_id}"
    average_distance = float(np.mean(distances)) if distances else 0.0

    return {
        "name": f"{export_file_name}.json",
        "scenario_id": scenario_id,
        "objects": objects,
        "roads": roads,
        "tl_states": tl_states,
        "metadata": {
            "sdc_track_index": sdc_indices[0],
            "log_name": scene.scene_metadata.log_name,
            "initial_lidar_timestamp": int(current_frame.timestamp),
            "map_name": scene.scene_metadata.map_name,
            "objects_of_interest": [],
            "tracks_to_predict": [],
            "average_distance_traveled": average_distance,
            "scenario_type": f"{scenario_type_prefix}+{scenario_id}",
        },
    }


def _load_navsim_scene(args: argparse.Namespace) -> Any:
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    scene_filter = SceneFilter(
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        frame_interval=args.frame_interval,
        has_route=not args.allow_missing_route,
        log_names=[args.log_name] if args.log_name else None,
        tokens=[args.token],
        include_synthetic_scenes=args.include_synthetic_scenes,
        synthetic_scene_tokens=[args.token] if args.include_synthetic_scenes else None,
    )
    loader = SceneLoader(
        data_path=args.navsim_log_path,
        original_sensor_path=args.original_sensor_path,
        scene_filter=scene_filter,
        synthetic_sensor_path=args.synthetic_sensor_path,
        synthetic_scenes_path=args.synthetic_scenes_path,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if args.token not in loader.tokens:
        raise ValueError(f"Token {args.token!r} was not found under {args.navsim_log_path}.")
    return loader.get_scene_from_token(args.token)


def _load_reference_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Reference JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _summary(scenario_json: dict[str, Any], reference_json: dict[str, Any] | None) -> dict[str, Any]:
    def _stats(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"n": 0, "mean": 0.0, "max": 0.0, "p95": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(len(array)),
            "mean": float(array.mean()),
            "max": float(array.max()),
            "p95": float(np.percentile(array, 95)),
        }

    def _position_array(obj: dict[str, Any], index: int) -> np.ndarray:
        position = obj["position"][index]
        return np.asarray([position["x"], position["y"], position["z"]], dtype=np.float64)

    def _velocity_array(obj: dict[str, Any], index: int) -> np.ndarray:
        velocity = obj["velocity"][index]
        return np.asarray([velocity["x"], velocity["y"]], dtype=np.float64)

    def _goal_array(obj: dict[str, Any]) -> np.ndarray:
        goal = obj["goalPosition"]
        return np.asarray([goal["x"], goal["y"], goal["z"]], dtype=np.float64)

    summary = {
        "name": scenario_json["name"],
        "scenario_id": scenario_json["scenario_id"],
        "objects": len(scenario_json["objects"]),
        "roads": len(scenario_json["roads"]),
        "tl_states": len(scenario_json["tl_states"]),
        "sdc_track_index": scenario_json["metadata"]["sdc_track_index"],
        "object_steps": len(scenario_json["objects"][0]["position"]) if scenario_json["objects"] else 0,
    }
    if reference_json is not None:
        reference_current_objects = [obj for obj in reference_json["objects"] if obj["valid"][0]]
        summary["reference"] = {
            "objects": len(reference_json["objects"]),
            "objects_valid_at_current_frame": len(reference_current_objects),
            "roads": len(reference_json["roads"]),
            "tl_states": len(reference_json["tl_states"]),
            "sdc_track_index": reference_json["metadata"].get("sdc_track_index"),
            "object_steps": len(reference_json["objects"][0]["position"]) if reference_json["objects"] else 0,
        }
        summary["delta"] = {
            "objects_vs_reference_all": summary["objects"] - summary["reference"]["objects"],
            "objects_vs_reference_current_valid": summary["objects"] - summary["reference"]["objects_valid_at_current_frame"],
            "roads": summary["roads"] - summary["reference"]["roads"],
            "tl_states": summary["tl_states"] - summary["reference"]["tl_states"],
        }
        if len(scenario_json["objects"]) == len(reference_current_objects):
            position_errors = []
            heading_errors = []
            velocity_errors = []
            speed_errors = []
            length_errors = []
            width_errors = []
            height_errors = []
            goal_vs_reference_goal_errors = []
            goal_vs_reference_4s_errors = []
            type_mismatches = []
            id_mismatches = []
            for obj, ref_obj in zip(scenario_json["objects"], reference_current_objects):
                if obj["type"] != ref_obj["type"] or bool(obj["is_sdc"]) != bool(ref_obj["is_sdc"]):
                    type_mismatches.append([obj.get("id"), ref_obj.get("id")])
                if obj.get("id") != ref_obj.get("id"):
                    id_mismatches.append([obj.get("id"), ref_obj.get("id")])
                obj_pos = _position_array(obj, 0)
                ref_pos = _position_array(ref_obj, 0)
                position_errors.append(float(np.linalg.norm(obj_pos[:2] - ref_pos[:2])))
                heading_errors.append(
                    abs(
                        float(
                            _wrap_yaw(
                                float(obj["heading"][0])
                                - float(ref_obj["heading"][0])
                            )
                        )
                    )
                )
                obj_velocity = _velocity_array(obj, 0)
                ref_velocity = _velocity_array(ref_obj, 0)
                velocity_errors.append(float(np.linalg.norm(obj_velocity - ref_velocity)))
                speed_errors.append(float(abs(np.linalg.norm(obj_velocity) - np.linalg.norm(ref_velocity))))
                length_errors.append(abs(float(obj["length"]) - float(ref_obj["length"])))
                width_errors.append(abs(float(obj["width"]) - float(ref_obj["width"])))
                height_errors.append(abs(float(obj["height"]) - float(ref_obj["height"])))
                goal_vs_reference_goal_errors.append(
                    float(np.linalg.norm(_goal_array(obj)[:2] - _goal_array(ref_obj)[:2]))
                )
                reference_4s_index = 40
                if len(ref_obj["position"]) > reference_4s_index and ref_obj["valid"][reference_4s_index]:
                    goal_vs_reference_4s_errors.append(
                        float(np.linalg.norm(_goal_array(obj)[:2] - _position_array(ref_obj, reference_4s_index)[:2]))
                    )
            summary["reference_current_frame_alignment"] = {
                "paired_objects": len(position_errors),
                "type_mismatches": type_mismatches,
                "id_mismatches": id_mismatches,
                "position_xy_error": _stats(position_errors),
                "heading_error": _stats(heading_errors),
                "velocity_xy_error": _stats(velocity_errors),
                "speed_error": _stats(speed_errors),
                "length_error": _stats(length_errors),
                "width_error": _stats(width_errors),
                "height_error": _stats(height_errors),
                "goal_error_vs_reference_final_goal": _stats(goal_vs_reference_goal_errors),
                "goal_error_vs_reference_4s_position": _stats(goal_vs_reference_4s_errors),
            }
        if len(scenario_json["roads"]) == len(reference_json["roads"]):
            road_point_errors = []
            road_type_mismatches = 0
            road_id_mismatches = 0
            road_geometry_count_mismatches = 0
            for road, ref_road in zip(scenario_json["roads"], reference_json["roads"]):
                if road["type"] != ref_road["type"] or road["map_element_id"] != ref_road["map_element_id"]:
                    road_type_mismatches += 1
                if road["id"] != ref_road["id"]:
                    road_id_mismatches += 1
                if len(road["geometry"]) != len(ref_road["geometry"]):
                    road_geometry_count_mismatches += 1
                for point, ref_point in zip(road["geometry"], ref_road["geometry"]):
                    road_point_errors.append(
                        float(
                            np.linalg.norm(
                                np.asarray([point["x"], point["y"], point["z"]], dtype=np.float64)
                                - np.asarray([ref_point["x"], ref_point["y"], ref_point["z"]], dtype=np.float64)
                            )
                        )
                    )
            summary["reference_road_alignment"] = {
                "type_mismatches": road_type_mismatches,
                "id_mismatches": road_id_mismatches,
                "geometry_count_mismatches": road_geometry_count_mismatches,
                "point_error": _stats(road_point_errors),
            }
    return summary


def _parse_args() -> argparse.Namespace:
    openscene_root = _default_openscene_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="NAVSIM current-frame token to export.")
    parser.add_argument("--log-name", default=DEFAULT_LOG_NAME, help="NAVSIM log name used to speed up lookup.")
    parser.add_argument("--navsim-log-path", type=Path, default=openscene_root / "navsim_logs" / "test")
    parser.add_argument("--original-sensor-path", type=Path, default=openscene_root / "sensor_blobs" / "test")
    parser.add_argument("--synthetic-scenes-path", type=Path, default=openscene_root / "navhard_two_stage" / "synthetic_scene_pickles")
    parser.add_argument("--synthetic-sensor-path", type=Path, default=openscene_root / "navhard_two_stage" / "sensor_blobs")
    parser.add_argument("--include-synthetic-scenes", action="store_true")
    parser.add_argument("--maps-root", type=Path, default=Path(os.environ.get("NUPLAN_MAPS_ROOT", openscene_root / "maps")))
    parser.add_argument("--output-dir", type=Path, default=_repo_root() / "outputs" / "gpudrive_json")
    parser.add_argument("--reference-json", type=Path, default=_workspace_root() / "ScenarioMax" / "tmp" / "token_obs_export" / "nuPlan_navsim_token_eval_00016f8b45c25a1d.json")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=8, help="Used only to load the NAVSIM hard_two_stage-shaped Scene; export uses the current frame only.")
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--allow-missing-route", action="store_true")
    parser.add_argument("--map-radius", type=int, default=DEFAULT_MAP_RADIUS_METERS)
    parser.add_argument(
        "--goal-source",
        choices=["auto", "future", "current"],
        default="auto",
        help="auto uses future frames for original/stage1 scenes and current position for synthetic/stage2 scenes.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--scenario-type-prefix", default=DEFAULT_SCENARIO_TYPE_PREFIX)
    parser.add_argument("--no-align-reference-order", action="store_true", help="Do not reorder objects to match --reference-json.")
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON instead of compact JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.navsim_log_path = args.navsim_log_path.expanduser().resolve()
    args.original_sensor_path = args.original_sensor_path.expanduser().resolve()
    args.synthetic_scenes_path = args.synthetic_scenes_path.expanduser().resolve()
    args.synthetic_sensor_path = args.synthetic_sensor_path.expanduser().resolve()
    args.maps_root = args.maps_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.reference_json = args.reference_json.expanduser().resolve() if args.reference_json else None

    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(args.navsim_log_path.parents[1]))
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.maps_root)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    reference_json = _load_reference_json(args.reference_json)
    scene = _load_navsim_scene(args)
    scenario_json = build_scenario_json(
        scene,
        map_radius=args.map_radius,
        goal_source=args.goal_source,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        scenario_type_prefix=args.scenario_type_prefix,
        reference_json=reference_json,
        align_reference_order=not args.no_align_reference_order,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / scenario_json["name"]
    with output_path.open("w", encoding="utf-8") as file:
        if args.pretty:
            json.dump(_jsonable(scenario_json), file, indent=2)
        else:
            json.dump(_jsonable(scenario_json), file)

    summary = _summary(scenario_json, reference_json)
    summary["json_path"] = str(output_path)
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
