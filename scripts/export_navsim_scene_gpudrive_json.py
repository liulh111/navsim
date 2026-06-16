#!/usr/bin/env python
"""Export one NAVSIM scene token to GPUDrive JSON and IDM route metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_TOKEN = "00016f8b45c25a1d"
DEFAULT_LOG_NAME = "2021.09.29.14.44.26_veh-28_00337_00504"
DEFAULT_DATASET_NAME = "nuPlan"
DEFAULT_DATASET_VERSION = "navsim_token_eval"
DEFAULT_SCENARIO_TYPE_PREFIX = "navsim_scene"
DEFAULT_MAP_RADIUS_METERS = 250
DEFAULT_ROUTE_SEARCH_DEPTH = 30
DEFAULT_ROUTE_GOAL_LOOKAHEAD_SECONDS = 4.0
DEFAULT_ROUTE_GOAL_MIN_DISTANCE_METERS = 20.0
DEFAULT_ROUTE_GOAL_MAX_DISTANCE_METERS = 40.0
DEFAULT_EXPORT_HORIZON_SECONDS = 4.0
DEFAULT_TARGET_INTERVAL_SECONDS = 0.1
DEFAULT_IDM_ROUTE_LENGTH_METERS = 80.0
DEFAULT_IDM_SNAP_THRESHOLD_METERS = 3.0
DEFAULT_STATIC_VEHICLE_SPEED_THRESHOLD_MPS = 0.1
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


@dataclass
class VehicleExportClassification:
    idm_routes: dict[str, dict[str, Any]]
    idm_vehicle_tokens: set[str]
    static_vehicle_tokens: set[str]
    dropped_vehicle_tokens: set[str]
    current_vehicle_tokens: set[str]


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


def _scene_is_synthetic(scene: Any) -> bool:
    metadata = scene.scene_metadata
    return bool(
        metadata.corresponding_original_scene
        or metadata.corresponding_original_initial_token
        or metadata.num_future_frames == 0
    )


def _target_times(horizon: float, interval: float) -> np.ndarray:
    if horizon <= 0 or interval <= 0:
        raise ValueError("horizon and target interval must be positive")
    num_steps = int(round(horizon / interval)) + 1
    if num_steps > 91:
        raise ValueError(f"GPUDrive supports at most 91 trajectory states, got {num_steps}")
    return np.arange(num_steps, dtype=np.float64) * interval


def _source_times(frames: list[Any], current_timestamp: int) -> np.ndarray:
    return np.asarray([(int(frame.timestamp) - current_timestamp) * 1e-6 for frame in frames], dtype=np.float64)


def _resample_agent_state(
    state: dict[str, np.ndarray],
    source_times: np.ndarray,
    target_times: np.ndarray,
    *,
    max_gap: float,
) -> dict[str, np.ndarray]:
    result = _empty_agent_state(len(target_times))
    valid_indices = np.flatnonzero(state["valid"].astype(bool))
    if len(valid_indices) == 0:
        return result

    valid_times = source_times[valid_indices]
    valid_headings = np.unwrap(state["heading"][valid_indices].astype(np.float64))
    exact_tolerance = 1e-4

    for target_index, target_time in enumerate(target_times):
        exact = np.flatnonzero(np.abs(valid_times - target_time) <= exact_tolerance)
        if len(exact):
            source_index = valid_indices[int(exact[0])]
            for key in ("position", "velocity", "length", "width", "height"):
                result[key][target_index] = state[key][source_index]
            result["heading"][target_index] = _wrap_yaw(state["heading"][source_index])
            result["valid"][target_index] = 1.0
            continue

        upper = int(np.searchsorted(valid_times, target_time, side="right"))
        if upper == 0 or upper >= len(valid_times):
            continue
        lower = upper - 1
        lower_time = float(valid_times[lower])
        upper_time = float(valid_times[upper])
        if upper_time - lower_time > max_gap + 1e-6:
            continue

        ratio = (target_time - lower_time) / (upper_time - lower_time)
        lower_index = valid_indices[lower]
        upper_index = valid_indices[upper]
        for key in ("position", "velocity", "length", "width", "height"):
            result[key][target_index] = (
                state[key][lower_index]
                + ratio * (state[key][upper_index] - state[key][lower_index])
            )
        result["heading"][target_index] = _wrap_yaw(
            valid_headings[lower] + ratio * (valid_headings[upper] - valid_headings[lower])
        )
        result["valid"][target_index] = 1.0

    return result


def _repeat_current_state(state: dict[str, np.ndarray], num_steps: int) -> dict[str, np.ndarray]:
    result = _empty_agent_state(num_steps)
    valid_indices = np.flatnonzero(state["valid"].astype(bool))
    if len(valid_indices) == 0:
        return result
    source_index = int(valid_indices[0])
    for key in ("position", "velocity", "length", "width", "height"):
        result[key][:] = state[key][source_index]
    result["heading"][:] = state["heading"][source_index]
    result["valid"][:] = 1.0
    return result


def _apply_vehicle_export_classification(
    dynamic_agents: dict[str, dict[str, Any]],
    classification: VehicleExportClassification,
    num_steps: int,
) -> dict[str, dict[str, Any]]:
    exported_vehicle_tokens = classification.idm_vehicle_tokens | classification.static_vehicle_tokens
    filtered_agents: dict[str, dict[str, Any]] = {}
    for token, agent in dynamic_agents.items():
        if token == "ego" or agent["type"] != VEHICLE:
            filtered_agents[token] = agent
            continue
        if token not in exported_vehicle_tokens:
            continue
        filtered_agents[token] = {
            **agent,
            "states": _repeat_current_state(agent["states"], num_steps),
            "mark_as_expert": False,
        }
    return filtered_agents


def _tracked_object_samples(
    scene: Any,
    center: np.ndarray,
    current_timestamp: int,
) -> dict[str, dict[str, Any]]:
    """Collect pedestrian/cyclist samples from current and extended detections."""
    from navsim.planning.scenario_builder.navsim_scenario_utils import (
        annotations_to_detection_tracks,
        ego_status_to_ego_state,
    )
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    current_index = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_index]
    ego_state = ego_status_to_ego_state(
        current_frame.ego_status,
        get_pacifica_parameters(),
        TimePoint(int(current_frame.timestamp)),
    )
    observations: list[tuple[float, list[Any]]] = [
        (
            0.0,
            list(
                annotations_to_detection_tracks(
                    current_frame.annotations, ego_state
                ).tracked_objects.tracked_objects
            ),
        )
    ]
    for detections in scene.extended_detections_tracks or []:
        objects = list(detections.tracked_objects.tracked_objects)
        timestamps = [int(obj.metadata.timestamp_us) for obj in objects]
        if not timestamps:
            continue
        observations.append(((min(timestamps) - current_timestamp) * 1e-6, objects))

    samples: dict[str, dict[str, Any]] = {}
    for time_s, objects in observations:
        for obj in objects:
            type_name = str(obj.tracked_object_type.name).lower()
            unified_type = NAVSIM_AGENT_TYPE_TO_UNIFIED.get(type_name)
            if unified_type not in {PEDESTRIAN, CYCLIST}:
                continue
            token = str(obj.track_token)
            entry = samples.setdefault(token, {"type": unified_type, "samples": []})
            velocity = getattr(obj, "velocity", None)
            entry["samples"].append(
                {
                    "time": float(time_s),
                    "position": np.asarray(
                        [obj.center.x - center[0], obj.center.y - center[1], 0.0],
                        dtype=np.float32,
                    ),
                    "heading": float(obj.center.heading),
                    "velocity": np.asarray(
                        [velocity.x, velocity.y] if velocity is not None else [0.0, 0.0],
                        dtype=np.float32,
                    ),
                    "length": float(obj.box.length),
                    "width": float(obj.box.width),
                    "height": float(obj.box.height),
                }
            )
    return samples


def _resample_vru_samples(
    samples: dict[str, dict[str, Any]],
    target_times: np.ndarray,
    source_interval: float,
) -> dict[str, dict[str, Any]]:
    agents: dict[str, dict[str, Any]] = {}
    for token, entry in samples.items():
        ordered = sorted(entry["samples"], key=lambda sample: sample["time"])
        deduplicated = []
        for sample in ordered:
            if deduplicated and math.isclose(sample["time"], deduplicated[-1]["time"], abs_tol=1e-4):
                deduplicated[-1] = sample
            else:
                deduplicated.append(sample)
        source_times = np.asarray([sample["time"] for sample in deduplicated], dtype=np.float64)
        state = _empty_agent_state(len(deduplicated))
        for index, sample in enumerate(deduplicated):
            for key in ("position", "velocity", "length", "width", "height"):
                state[key][index] = sample[key]
            state["heading"][index] = sample["heading"]
            state["valid"][index] = 1.0
        agents[token] = {
            "type": entry["type"],
            "states": _resample_agent_state(
                state,
                source_times,
                target_times,
                max_gap=source_interval * 1.1,
            ),
            "mark_as_expert": True,
        }
    return agents


def build_dynamic_agents(
    scene: Any,
    center: np.ndarray,
    *,
    horizon: float,
    target_interval: float,
) -> tuple[dict[str, dict[str, Any]], np.ndarray, str]:
    current_index = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_index]
    target_times = _target_times(horizon, target_interval)
    synthetic = _scene_is_synthetic(scene)

    if synthetic:
        current_agents = _extract_source_dynamic_agents([current_frame], center)
        dynamic_agents = {}
        for token, agent in current_agents.items():
            if agent["type"] in {PEDESTRIAN, CYCLIST}:
                continue
            dynamic_agents[token] = {
                **agent,
                "states": _repeat_current_state(agent["states"], len(target_times)),
                "mark_as_expert": False,
            }
        dynamic_agents.update(
            _resample_vru_samples(
                _tracked_object_samples(scene, center, int(current_frame.timestamp)),
                target_times,
                source_interval=0.5,
            )
        )
        return dynamic_agents, target_times, "synthetic_current_plus_extended_vru"

    future_end = min(
        len(scene.frames),
        current_index + int(round(horizon / 0.5)) + 1,
    )
    source_frames = scene.frames[current_index:future_end]
    source_agents = _extract_source_dynamic_agents(source_frames, center)
    source_times = _source_times(source_frames, int(current_frame.timestamp))
    source_interval = float(np.median(np.diff(source_times))) if len(source_times) > 1 else 0.5
    dynamic_agents = {}
    for token, agent in source_agents.items():
        dynamic_agents[token] = {
            **agent,
            "states": _resample_agent_state(
                agent["states"],
                source_times,
                target_times,
                max_gap=source_interval * 1.1,
            ),
            "mark_as_expert": agent["type"] in {PEDESTRIAN, CYCLIST},
        }
    return dynamic_agents, target_times, "navsim_future_log"


def _append_route_points(points: list[np.ndarray], lane: Any) -> None:
    for pose in lane.baseline_path.discrete_path:
        point = np.asarray([pose.x, pose.y], dtype=np.float64)
        if points and np.linalg.norm(point - points[-1]) < 1e-3:
            continue
        points.append(point)


def _build_vehicle_route(
    vehicle: Any,
    map_api: Any,
    *,
    minimum_length: float,
    snap_threshold: float,
) -> dict[str, Any] | None:
    from nuplan.planning.simulation.observation.idm.idm_agents_builder import get_starting_segment

    lane, initial_progress = get_starting_segment(vehicle, map_api)
    if lane is None or initial_progress is None:
        return None
    snapped_pose = lane.baseline_path.get_nearest_pose_from_position(vehicle.center.point)
    snap_distance = float(
        np.hypot(snapped_pose.x - vehicle.center.x, snapped_pose.y - vehicle.center.y)
    )
    if snap_distance > snap_threshold:
        return None

    route_lanes = [lane]
    points: list[np.ndarray] = []
    _append_route_points(points, lane)
    accumulated = _polyline_cumulative_lengths(np.asarray(points, dtype=np.float64))
    remaining = float(accumulated[-1] - initial_progress) if len(accumulated) else 0.0
    while remaining < minimum_length and len(route_lanes) < 100:
        outgoing = list(route_lanes[-1].outgoing_edges)
        if not outgoing:
            break
        next_lane = min(
            outgoing,
            key=lambda edge: abs(edge.baseline_path.get_curvature_at_arc_length(0.0)),
        )
        route_lanes.append(next_lane)
        _append_route_points(points, next_lane)
        accumulated = _polyline_cumulative_lengths(np.asarray(points, dtype=np.float64))
        remaining = float(accumulated[-1] - initial_progress)

    route_xy = np.asarray(points, dtype=np.float64)
    if len(route_xy) < 2:
        return None
    progress = _polyline_cumulative_lengths(route_xy)
    deltas = np.diff(route_xy, axis=0)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    headings = np.concatenate([headings, headings[-1:]])
    return {
        "track_token": str(vehicle.track_token),
        "route_xy_global": route_xy,
        "route_heading": headings,
        "route_progress": progress,
        "initial_progress": float(initial_progress),
        "snap_distance": snap_distance,
        "snapped_pose": np.asarray(
            [snapped_pose.x, snapped_pose.y, snapped_pose.heading], dtype=np.float64
        ),
        "lane_ids": [str(route_lane.id) for route_lane in route_lanes],
    }


def classify_vehicle_export_tokens(
    scene: Any,
    center: np.ndarray,
    *,
    minimum_length: float = DEFAULT_IDM_ROUTE_LENGTH_METERS,
    snap_threshold: float = DEFAULT_IDM_SNAP_THRESHOLD_METERS,
    static_speed_threshold: float = DEFAULT_STATIC_VEHICLE_SPEED_THRESHOLD_MPS,
) -> VehicleExportClassification:
    from navsim.planning.scenario_builder.navsim_scenario_utils import (
        annotations_to_detection_tracks,
        ego_status_to_ego_state,
    )
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    current_index = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_index]
    ego_state = ego_status_to_ego_state(
        current_frame.ego_status,
        get_pacifica_parameters(),
        TimePoint(int(current_frame.timestamp)),
    )
    vehicles = annotations_to_detection_tracks(
        current_frame.annotations, ego_state
    ).tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)

    routes: dict[str, dict[str, Any]] = {}
    idm_vehicle_tokens: set[str] = set()
    current_vehicle_tokens = {str(vehicle.track_token) for vehicle in vehicles}
    occupied_geometries = [ego_state.agent.box.geometry]
    active_geometries: list[Any] = []
    inactive_vehicles: list[tuple[Any, dict[str, Any] | None]] = []

    for vehicle in vehicles:
        route = _build_vehicle_route(
            vehicle,
            scene.map_api,
            minimum_length=minimum_length,
            snap_threshold=snap_threshold,
        )
        if route is None:
            inactive_vehicles.append((vehicle, None))
            continue
        snapped_box = OrientedBox.from_new_pose(
            vehicle.box,
            StateSE2(*route["snapped_pose"]),
        )
        if any(geometry.intersects(snapped_box.geometry) for geometry in occupied_geometries):
            inactive_vehicles.append((vehicle, route))
            continue
        occupied_geometries.append(snapped_box.geometry)
        active_geometries.append(snapped_box.geometry)
        route["route_xy"] = route.pop("route_xy_global") - center[None, :]
        token = str(vehicle.track_token)
        routes[token] = route
        idm_vehicle_tokens.add(token)

    static_vehicle_tokens: set[str] = set()
    for vehicle, route in inactive_vehicles:
        token = str(vehicle.track_token)
        collides_with_active = any(vehicle.box.geometry.intersects(geometry) for geometry in active_geometries)
        if collides_with_active:
            continue

        is_stationary = float(vehicle.velocity.magnitude()) < static_speed_threshold
        is_in_lanes = scene.map_api.is_in_layer(vehicle.center, SemanticMapLayer.LANE) or scene.map_api.is_in_layer(
            vehicle.center, SemanticMapLayer.INTERSECTION
        )

        lateral_deviation = None
        if route is not None:
            lateral_deviation = route["snap_distance"]
        else:
            from nuplan.planning.simulation.observation.idm.idm_agents_builder import get_starting_segment

            lane, _ = get_starting_segment(vehicle, scene.map_api)
            if lane is not None:
                state_on_path = lane.baseline_path.get_nearest_pose_from_position(vehicle.center)
                lateral_deviation = float(
                    np.hypot(state_on_path.x - vehicle.center.x, state_on_path.y - vehicle.center.y)
                )

        if is_stationary and not is_in_lanes:
            static_vehicle_tokens.add(token)
            continue
        if lateral_deviation is not None and lateral_deviation > snap_threshold:
            static_vehicle_tokens.add(token)

    exported_vehicle_tokens = idm_vehicle_tokens | static_vehicle_tokens
    dropped_vehicle_tokens = current_vehicle_tokens - exported_vehicle_tokens
    return VehicleExportClassification(
        idm_routes=routes,
        idm_vehicle_tokens=idm_vehicle_tokens,
        static_vehicle_tokens=static_vehicle_tokens,
        dropped_vehicle_tokens=dropped_vehicle_tokens,
        current_vehicle_tokens=current_vehicle_tokens,
    )


def _load_route_dicts(map_api: Any, route_roadblock_ids: Iterable[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    unique_route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))
    route_roadblock_dict: dict[str, Any] = {}
    route_lane_dict: dict[str, Any] = {}

    for roadblock_id in unique_route_roadblock_ids:
        roadblock = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
        roadblock = roadblock or map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if roadblock is None:
            continue

        route_roadblock_dict[roadblock.id] = roadblock
        for lane in roadblock.interior_edges:
            route_lane_dict[lane.id] = lane

    return route_roadblock_dict, route_lane_dict


def _find_starting_lane(route_lane_dict: dict[str, Any], ego_state: Any) -> Any | None:
    if not route_lane_dict:
        return None

    ego_xy = ego_state.rear_axle.point.array
    ego_heading = ego_state.rear_axle.heading
    best_lane = None
    best_score = math.inf

    for lane in route_lane_dict.values():
        discrete_path = getattr(lane.baseline_path, "discrete_path", None)
        if not discrete_path:
            continue

        lane_xy = np.asarray([state.point.array for state in discrete_path], dtype=np.float64)
        distances = np.linalg.norm(lane_xy - ego_xy[None, :], axis=1)
        nearest_index = int(np.argmin(distances))
        heading_error = abs(float(_wrap_yaw(discrete_path[nearest_index].heading - ego_heading)))
        score = float(distances[nearest_index] + 2.0 * heading_error)

        if score < best_score:
            best_score = score
            best_lane = lane

    return best_lane


def _polyline_cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(points) == 1:
        return np.zeros((1,), dtype=np.float64)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(segment_lengths, dtype=np.float64)])


def _project_onto_polyline(points: np.ndarray, query_xy: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) == 0:
        raise ValueError("Polyline must contain at least one point.")
    if len(points) == 1:
        return np.asarray(points[0], dtype=np.float64), 0.0

    cumulative_lengths = _polyline_cumulative_lengths(points)
    best_distance = math.inf
    best_projection = np.asarray(points[0], dtype=np.float64)
    best_progress = 0.0

    for index in range(len(points) - 1):
        start_xy = np.asarray(points[index], dtype=np.float64)
        end_xy = np.asarray(points[index + 1], dtype=np.float64)
        segment = end_xy - start_xy
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-6:
            continue

        direction = segment / segment_length
        local_t = float(np.dot(query_xy - start_xy, direction))
        local_t = float(np.clip(local_t, 0.0, segment_length))
        projection = start_xy + direction * local_t
        distance = float(np.linalg.norm(query_xy - projection))

        if distance < best_distance:
            best_distance = distance
            best_projection = projection
            best_progress = float(cumulative_lengths[index] + local_t)

    return best_projection, best_progress


def _interpolate_polyline_at_progress(points: np.ndarray, progress: float) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("Polyline must contain at least one point.")
    if len(points) == 1:
        return np.asarray(points[0], dtype=np.float64)

    cumulative_lengths = _polyline_cumulative_lengths(points)
    clamped_progress = float(np.clip(progress, 0.0, cumulative_lengths[-1]))
    upper_index = int(np.searchsorted(cumulative_lengths, clamped_progress, side="right"))
    upper_index = min(max(upper_index, 1), len(points) - 1)
    lower_index = upper_index - 1
    lower_progress = float(cumulative_lengths[lower_index])
    upper_progress = float(cumulative_lengths[upper_index])

    if upper_progress - lower_progress < 1e-6:
        return np.asarray(points[upper_index], dtype=np.float64)

    ratio = (clamped_progress - lower_progress) / (upper_progress - lower_progress)
    return np.asarray(points[lower_index], dtype=np.float64) + ratio * (
        np.asarray(points[upper_index], dtype=np.float64) - np.asarray(points[lower_index], dtype=np.float64)
    )


def _compute_route_goal_lookahead_distance(ego_speed_mps: float) -> float:
    horizon_distance = max(0.0, ego_speed_mps) * DEFAULT_ROUTE_GOAL_LOOKAHEAD_SECONDS
    return float(
        np.clip(
            horizon_distance,
            DEFAULT_ROUTE_GOAL_MIN_DISTANCE_METERS,
            DEFAULT_ROUTE_GOAL_MAX_DISTANCE_METERS,
        )
    )


def _build_route_centerline_points(scene: Any, ego_state: Any, route_roadblock_ids: Iterable[str]) -> np.ndarray | None:
    from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
    from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction

    if scene.map_api is None:
        return None

    route_roadblock_dict, route_lane_dict = _load_route_dicts(scene.map_api, route_roadblock_ids)
    if not route_roadblock_dict or not route_lane_dict:
        return None

    corrected_route_ids = route_roadblock_correction(ego_state.rear_axle, scene.map_api, route_roadblock_dict)
    route_roadblock_dict, route_lane_dict = _load_route_dicts(scene.map_api, corrected_route_ids)
    if not route_roadblock_dict or not route_lane_dict:
        return None

    starting_lane = _find_starting_lane(route_lane_dict, ego_state)
    if starting_lane is None:
        return None

    route_roadblocks = list(route_roadblock_dict.values())
    route_roadblock_ids = list(route_roadblock_dict.keys())
    start_roadblock_id = starting_lane.get_roadblock_id()
    start_index = route_roadblock_ids.index(start_roadblock_id) if start_roadblock_id in route_roadblock_ids else 0
    target_index = min(len(route_roadblocks) - 1, start_index + DEFAULT_ROUTE_SEARCH_DEPTH - 1)
    target_roadblock = route_roadblocks[target_index]

    graph_search = Dijkstra(starting_lane, list(route_lane_dict.keys()))
    route_plan, path_found = graph_search.search(target_roadblock)
    if not path_found or not route_plan:
        route_plan = [starting_lane]

    centerline_points: list[np.ndarray] = []
    for lane in route_plan:
        for state in lane.baseline_path.discrete_path:
            point = np.asarray(state.point.array, dtype=np.float64)
            if centerline_points and np.linalg.norm(point - centerline_points[-1]) < 1e-3:
                continue
            centerline_points.append(point)

    if not centerline_points:
        return None
    return np.asarray(centerline_points, dtype=np.float64)


def _extract_route_terminal_goal_position(scene: Any, current_frame_idx: int, center: np.ndarray) -> np.ndarray | None:
    from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    if scene.map_api is None:
        return None

    current_frame = scene.frames[current_frame_idx]
    if not current_frame.roadblock_ids:
        return None

    ego_state = ego_status_to_ego_state(
        current_frame.ego_status,
        get_pacifica_parameters(),
        TimePoint(int(current_frame.timestamp)),
    )
    centerline_points = _build_route_centerline_points(scene, ego_state, current_frame.roadblock_ids)
    if centerline_points is None or len(centerline_points) == 0:
        return None

    goal_xy = centerline_points[-1]
    return np.asarray([goal_xy[0] - center[0], goal_xy[1] - center[1], 0.0], dtype=np.float32)


def _extract_route_local_goal_position(scene: Any, current_frame_idx: int, center: np.ndarray) -> np.ndarray | None:
    from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    if scene.map_api is None:
        return None

    current_frame = scene.frames[current_frame_idx]
    if not current_frame.roadblock_ids:
        return None

    ego_state = ego_status_to_ego_state(
        current_frame.ego_status,
        get_pacifica_parameters(),
        TimePoint(int(current_frame.timestamp)),
    )
    centerline_points = _build_route_centerline_points(scene, ego_state, current_frame.roadblock_ids)
    if centerline_points is None or len(centerline_points) == 0:
        return None

    ego_xy = np.asarray(ego_state.rear_axle.point.array, dtype=np.float64)
    _, current_progress = _project_onto_polyline(centerline_points, ego_xy)
    ego_speed_mps = float(np.linalg.norm(np.asarray(current_frame.ego_status.ego_velocity, dtype=np.float64)))
    target_progress = current_progress + _compute_route_goal_lookahead_distance(ego_speed_mps)
    goal_xy = _interpolate_polyline_at_progress(centerline_points, target_progress)
    return np.asarray([goal_xy[0] - center[0], goal_xy[1] - center[1], 0.0], dtype=np.float32)


def _resolve_goal_source(scene: Any, goal_source: str) -> str:
    if goal_source in {"current", "future", "route_terminal", "route_local"}:
        return goal_source
    return "route_local" if bool(
        scene.scene_metadata.corresponding_original_scene
        or scene.scene_metadata.corresponding_original_initial_token
    ) else "future"


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

    resolved_goal_source = _resolve_goal_source(scene, goal_source)
    route_goal_position = None
    if resolved_goal_source == "route_terminal":
        route_goal_position = _extract_route_terminal_goal_position(scene, current_frame_idx, center)
        resolved_goal_source = "current"
    elif resolved_goal_source == "route_local":
        route_goal_position = _extract_route_local_goal_position(scene, current_frame_idx, center)
        resolved_goal_source = "current"

    end_frame_idx = current_frame_idx if resolved_goal_source == "current" else len(scene.frames) - 1
    end_frame_idx = max(current_frame_idx, min(end_frame_idx, len(scene.frames) - 1))
    vehicle_parameters = get_pacifica_parameters()
    goal_positions: dict[str, np.ndarray] = {}

    has_route_goal = route_goal_position is not None
    if has_route_goal:
        goal_positions["ego"] = route_goal_position

    for frame_idx in range(current_frame_idx, end_frame_idx + 1):
        frame = scene.frames[frame_idx]
        ego_state = ego_status_to_ego_state(
            frame.ego_status,
            vehicle_parameters,
            TimePoint(int(frame.timestamp)),
        )
        if not has_route_goal and (resolved_goal_source == "future" or "ego" not in goal_positions):
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
        "mark_as_expert": bool(agent.get("mark_as_expert", False)),
        "total_distance_traveled": _object_distance_traveled(positions, valids),
        "_track_token": object_id,
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


def _strip_internal_object_fields(objects: list[dict[str, Any]]) -> None:
    for obj in objects:
        obj.pop("_track_token", None)


def validate_export_artifacts(
    scenario_json: dict[str, Any],
    route_sidecar: dict[str, Any],
    expected_steps: int,
) -> None:
    objects = scenario_json["objects"]
    object_ids = {int(obj["id"]) for obj in objects}
    for obj in objects:
        lengths = {
            len(obj["position"]),
            len(obj["heading"]),
            len(obj["velocity"]),
            len(obj["valid"]),
        }
        if lengths != {expected_steps}:
            raise ValueError(
                f"Agent {obj['id']} has inconsistent trajectory lengths {sorted(lengths)}; "
                f"expected {expected_steps}"
            )
    route_ids = {int(agent_id) for agent_id in route_sidecar["vehicles"]}
    if not route_ids.issubset(object_ids):
        raise ValueError(f"IDM routes reference missing GPUDrive agent IDs: {sorted(route_ids - object_ids)}")
    if len(route_ids) != len(route_sidecar["vehicles"]):
        raise ValueError("IDM route sidecar contains duplicate agent IDs")


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


def build_export_artifacts(
    scene: Any,
    *,
    map_radius: int,
    goal_source: str,
    dataset_name: str,
    dataset_version: str,
    scenario_type_prefix: str,
    reference_json: dict[str, Any] | None,
    align_reference_order: bool,
    target_interval: float = DEFAULT_TARGET_INTERVAL_SECONDS,
    horizon: float = DEFAULT_EXPORT_HORIZON_SECONDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    current_frame_idx = scene.scene_metadata.num_history_frames - 1
    if current_frame_idx >= len(scene.frames):
        raise ValueError("The NAVSIM scene has no frames from the current frame onward.")
    current_frame = scene.frames[current_frame_idx]
    vehicle_parameters = get_pacifica_parameters()
    current_ego_state = ego_status_to_ego_state(
        current_frame.ego_status,
        vehicle_parameters,
        TimePoint(int(current_frame.timestamp)),
    )
    center = np.asarray([current_ego_state.waypoint.x, current_ego_state.waypoint.y], dtype=np.float64)
    dynamic_agents, target_times, trajectory_source = build_dynamic_agents(
        scene,
        center,
        horizon=horizon,
        target_interval=target_interval,
    )
    goal_positions = extract_goal_positions(scene, current_frame_idx, center, goal_source)
    for object_id, agent in dynamic_agents.items():
        if object_id in goal_positions:
            agent["goal_position"] = goal_positions[object_id]

    vehicle_classification = classify_vehicle_export_tokens(scene, center)
    dynamic_agents = _apply_vehicle_export_classification(
        dynamic_agents,
        vehicle_classification,
        len(target_times),
    )
    idm_routes = vehicle_classification.idm_routes

    static_map_elements = extract_static_map_elements(scene.map_api, center, map_radius)
    roads, _ = convert_map_features(static_map_elements)
    traffic_light_frames = scene.frames[current_frame_idx : current_frame_idx + len(target_times)]
    if not traffic_light_frames:
        traffic_light_frames = [current_frame]
    tl_states = convert_traffic_lights(
        extract_dynamic_map_elements(
            traffic_light_frames,
            scene.map_api,
            center,
            target_times,
        )
    )
    objects, distances = convert_dynamic_agents(dynamic_agents)

    if reference_json is not None and align_reference_order:
        objects = align_object_order_to_reference(objects, reference_json.get("objects", []))

    sdc_indices = [index for index, obj in enumerate(objects) if obj["is_sdc"]]
    if not sdc_indices:
        raise ValueError("No ego/SDC object was exported.")

    scenario_id = scene.scene_metadata.initial_token
    export_file_name = f"{dataset_name}_{dataset_version}_{scenario_id}"
    average_distance = float(np.mean(distances)) if distances else 0.0

    scenario_json = {
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
            "source": trajectory_source,
            "target_interval": float(target_interval),
            "horizon": float(horizon),
        },
    }
    route_entries: dict[str, dict[str, Any]] = {}
    for obj in objects:
        track_token = str(obj.get("_track_token", ""))
        if track_token not in idm_routes:
            continue
        route = idm_routes[track_token]
        route_entries[str(obj["id"])] = {
            "agent_id": int(obj["id"]),
            "track_token": track_token,
            "route_xy": route["route_xy"],
            "route_heading": route["route_heading"],
            "route_progress": route["route_progress"],
            "initial_progress": route["initial_progress"],
            "snap_distance": route["snap_distance"],
            "lane_ids": route["lane_ids"],
        }

    route_sidecar = {
        "scene_id": scenario_id,
        "coordinate_frame": "gpudrive_json",
        "target_interval": float(target_interval),
        "horizon": float(horizon),
        "minimum_route_length": DEFAULT_IDM_ROUTE_LENGTH_METERS,
        "snap_threshold": DEFAULT_IDM_SNAP_THRESHOLD_METERS,
        "current_vehicle_count": len(vehicle_classification.current_vehicle_tokens),
        "idm_vehicle_count": len(vehicle_classification.idm_vehicle_tokens),
        "static_vehicle_count": len(vehicle_classification.static_vehicle_tokens),
        "dropped_vehicle_count": len(vehicle_classification.dropped_vehicle_tokens),
        "filtered_vehicle_count": len(vehicle_classification.dropped_vehicle_tokens),
        "current_vehicle_track_tokens": sorted(vehicle_classification.current_vehicle_tokens),
        "idm_vehicle_track_tokens": sorted(vehicle_classification.idm_vehicle_tokens),
        "static_vehicle_track_tokens": sorted(vehicle_classification.static_vehicle_tokens),
        "dropped_vehicle_track_tokens": sorted(vehicle_classification.dropped_vehicle_tokens),
        "vehicles": route_entries,
    }
    _strip_internal_object_fields(objects)
    validate_export_artifacts(scenario_json, route_sidecar, len(target_times))
    return scenario_json, route_sidecar


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
    target_interval: float = DEFAULT_TARGET_INTERVAL_SECONDS,
    horizon: float = DEFAULT_EXPORT_HORIZON_SECONDS,
) -> dict[str, Any]:
    scenario_json, _ = build_export_artifacts(
        scene,
        map_radius=map_radius,
        goal_source=goal_source,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        scenario_type_prefix=scenario_type_prefix,
        reference_json=reference_json,
        align_reference_order=align_reference_order,
        target_interval=target_interval,
        horizon=horizon,
    )
    return scenario_json


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
    parser.add_argument("--target-interval", type=float, default=DEFAULT_TARGET_INTERVAL_SECONDS)
    parser.add_argument("--horizon", type=float, default=DEFAULT_EXPORT_HORIZON_SECONDS)
    parser.add_argument(
        "--goal-source",
        choices=["auto", "future", "current", "route_terminal", "route_local"],
        default="auto",
        help="auto uses future frames for original/stage1 scenes and route-local goal for synthetic/stage2 scenes.",
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

    reference_json = (
        _load_reference_json(args.reference_json)
        if not args.no_align_reference_order
        else None
    )
    scene = _load_navsim_scene(args)
    scenario_json, route_sidecar = build_export_artifacts(
        scene,
        map_radius=args.map_radius,
        goal_source=args.goal_source,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        scenario_type_prefix=args.scenario_type_prefix,
        reference_json=reference_json,
        align_reference_order=not args.no_align_reference_order,
        target_interval=args.target_interval,
        horizon=args.horizon,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / scenario_json["name"]
    with output_path.open("w", encoding="utf-8") as file:
        if args.pretty:
            json.dump(_jsonable(scenario_json), file, indent=2)
        else:
            json.dump(_jsonable(scenario_json), file)
    route_path = output_path.with_suffix(".idm_routes.json")
    with route_path.open("w", encoding="utf-8") as file:
        if args.pretty:
            json.dump(_jsonable(route_sidecar), file, indent=2)
        else:
            json.dump(_jsonable(route_sidecar), file)

    summary = _summary(scenario_json, reference_json)
    summary["json_path"] = str(output_path)
    summary["route_sidecar_path"] = str(route_path)
    summary["idm_vehicles"] = len(route_sidecar["vehicles"])
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
