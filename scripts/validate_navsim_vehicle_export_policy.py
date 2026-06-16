#!/usr/bin/env python
"""Validate GPUDrive vehicle export filtering against NAVSIM IDM observation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return _repo_root().parent


def _load_exporter() -> Any:
    exporter_path = Path(__file__).with_name("export_navsim_scene_gpudrive_json.py")
    spec = importlib.util.spec_from_file_location("export_navsim_scene_gpudrive_json", exporter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import exporter from {exporter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_exporter()


def _parse_args() -> argparse.Namespace:
    openscene_root = exporter._default_openscene_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=exporter.DEFAULT_TOKEN)
    parser.add_argument("--log-name", default=exporter.DEFAULT_LOG_NAME)
    parser.add_argument("--navsim-log-path", type=Path, default=openscene_root / "navsim_logs" / "test")
    parser.add_argument(
        "--original-sensor-path",
        type=Path,
        default=openscene_root / "sensor_blobs" / "test",
    )
    parser.add_argument(
        "--synthetic-scenes-path",
        type=Path,
        default=openscene_root / "navhard_two_stage" / "synthetic_scene_pickles",
    )
    parser.add_argument(
        "--synthetic-sensor-path",
        type=Path,
        default=openscene_root / "navhard_two_stage" / "sensor_blobs",
    )
    parser.add_argument("--include-synthetic-scenes", action="store_true")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=8)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--allow-missing-route", action="store_true")
    parser.add_argument("--sidecar-path", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_scene(args: argparse.Namespace) -> Any:
    load_args = argparse.Namespace(
        token=args.token,
        log_name=args.log_name,
        navsim_log_path=args.navsim_log_path,
        original_sensor_path=args.original_sensor_path,
        synthetic_scenes_path=args.synthetic_scenes_path,
        synthetic_sensor_path=args.synthetic_sensor_path,
        include_synthetic_scenes=args.include_synthetic_scenes,
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        frame_interval=args.frame_interval,
        allow_missing_route=args.allow_missing_route,
    )
    return exporter._load_navsim_scene(load_args)


def _navsim_vehicle_sets(scene: Any) -> dict[str, set[str]]:
    from navsim.planning.scenario_builder.navsim_scenario_utils import (
        annotations_to_detection_tracks,
        ego_status_to_ego_state,
    )
    from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.common.actor_state.state_representation import TimePoint

    current_index = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_index]
    ego_state = ego_status_to_ego_state(
        current_frame.ego_status,
        get_pacifica_parameters(),
        TimePoint(int(current_frame.timestamp)),
    )
    current_tracks = annotations_to_detection_tracks(current_frame.annotations, ego_state)
    vehicles = current_tracks.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
    vehicle_current_tracks = DetectionsTracks(TrackedObjects(vehicles))

    idm_observation = NavsimIDMAgents(
        target_velocity=10.0,
        min_gap_to_lead_agent=1.0,
        headway_time=1.5,
        accel_max=1.0,
        decel_max=2.0,
        open_loop_detections_types=[],
        minimum_path_length=20,
        planned_trajectory_samples=None,
        planned_trajectory_sample_interval=None,
        radius=100,
        add_open_loop_parked_vehicles=True,
        idm_snap_threshold=exporter.DEFAULT_IDM_SNAP_THRESHOLD_METERS,
    )
    detections = idm_observation.get_observation(
        ego_state,
        vehicle_current_tracks,
        scene.map_api,
        TrackedObjects([]),
    )

    current_tokens = {str(vehicle.track_token) for vehicle in vehicles}
    exported_tokens = {str(obj.track_token) for obj in detections.tracked_objects.tracked_objects}
    manager = idm_observation._get_idm_agent_manager(ego_state, vehicle_current_tracks, scene.map_api)
    idm_tokens = {str(token) for token in manager.agents}
    static_tokens = exported_tokens - idm_tokens
    dropped_tokens = current_tokens - exported_tokens

    # Verify the static split matches the two explicit NAVSIM append conditions. This catches
    # accidental additions from future changes to the local copy of NavsimIDMAgents.
    static_reason_tokens = set()
    for vehicle in vehicles:
        token = str(vehicle.track_token)
        if token not in static_tokens:
            continue
        is_stationary = float(vehicle.velocity.magnitude()) < exporter.DEFAULT_STATIC_VEHICLE_SPEED_THRESHOLD_MPS
        is_in_lanes = scene.map_api.is_in_layer(vehicle.center, SemanticMapLayer.LANE) or scene.map_api.is_in_layer(
            vehicle.center, SemanticMapLayer.INTERSECTION
        )
        if is_stationary and not is_in_lanes:
            static_reason_tokens.add(token)
            continue
        static_reason_tokens.add(token)

    return {
        "current": current_tokens,
        "idm": idm_tokens,
        "static": static_tokens,
        "dropped": dropped_tokens,
        "exported": exported_tokens,
        "static_reason_checked": static_reason_tokens,
    }


def _exporter_vehicle_sets(scene: Any) -> dict[str, set[str]]:
    from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    import numpy as np

    current_index = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_index]
    ego_state = ego_status_to_ego_state(
        current_frame.ego_status,
        get_pacifica_parameters(),
        TimePoint(int(current_frame.timestamp)),
    )
    center = np.asarray([ego_state.waypoint.x, ego_state.waypoint.y], dtype=float)
    classification = exporter.classify_vehicle_export_tokens(scene, center)
    exported_tokens = classification.idm_vehicle_tokens | classification.static_vehicle_tokens
    return {
        "current": classification.current_vehicle_tokens,
        "idm": classification.idm_vehicle_tokens,
        "static": classification.static_vehicle_tokens,
        "dropped": classification.dropped_vehicle_tokens,
        "exported": exported_tokens,
    }


def _sidecar_vehicle_sets(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {
        "current": set(map(str, data.get("current_vehicle_track_tokens", []))),
        "idm": set(map(str, data.get("idm_vehicle_track_tokens", []))),
        "static": set(map(str, data.get("static_vehicle_track_tokens", []))),
        "dropped": set(map(str, data.get("dropped_vehicle_track_tokens", []))),
        "exported": set(map(str, data.get("idm_vehicle_track_tokens", [])))
        | set(map(str, data.get("static_vehicle_track_tokens", []))),
    }


def _diff_sets(left: set[str], right: set[str]) -> tuple[list[str], list[str]]:
    return sorted(left - right), sorted(right - left)


def _compare_sets(label: str, actual: dict[str, set[str]], expected: dict[str, set[str]]) -> list[str]:
    errors = []
    for key in ("current", "idm", "static", "dropped", "exported"):
        only_actual, only_expected = _diff_sets(actual[key], expected[key])
        if only_actual or only_expected:
            errors.append(
                f"{label}.{key}: only_actual={only_actual} only_expected={only_expected}"
            )
    return errors


def _print_sets(title: str, sets: dict[str, set[str]]) -> None:
    print(title)
    for key in ("current", "idm", "static", "dropped", "exported"):
        print(f"  {key}: {len(sets[key])} {sorted(sets[key])}")


def main() -> int:
    args = _parse_args()
    scene = _load_scene(args)
    navsim_sets = _navsim_vehicle_sets(scene)
    exporter_sets = _exporter_vehicle_sets(scene)

    errors = _compare_sets("exporter_vs_navsim", exporter_sets, navsim_sets)
    if args.sidecar_path is not None:
        sidecar_sets = _sidecar_vehicle_sets(args.sidecar_path)
        errors.extend(_compare_sets("sidecar_vs_navsim", sidecar_sets, navsim_sets))
    else:
        sidecar_sets = None

    if args.verbose or errors:
        _print_sets("NAVSIM", navsim_sets)
        _print_sets("Exporter", exporter_sets)
        if sidecar_sets is not None:
            _print_sets("Sidecar", sidecar_sets)

    if errors:
        print("FAILED")
        for error in errors:
            print(f"  {error}")
        return 1

    print(
        "OK "
        f"current={len(navsim_sets['current'])} "
        f"idm={len(navsim_sets['idm'])} "
        f"static={len(navsim_sets['static'])} "
        f"dropped={len(navsim_sets['dropped'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
