#!/usr/bin/env python
"""Inspect future tracked objects stored in a NAVSIM synthetic scene pickle."""

from __future__ import annotations

import argparse
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("OPENSCENE_DATA_ROOT", WORKSPACE_ROOT / "dataset"))
DEFAULT_SCENE_DIR = DEFAULT_DATA_ROOT / "navhard_two_stage" / "synthetic_scene_pickles"
DEFAULT_EXPECTED_INTERVAL_S = 0.5
DEFAULT_EXPECTED_HORIZON_S = 4.0
DEFAULT_MAX_STEP_DISTANCE_M = 20.0

TYPE_ALIASES = {
    "cyclist": "BICYCLE",
    "bike": "BICYCLE",
}
DEFAULT_CONTINUITY_TYPES = {"VEHICLE", "PEDESTRIAN", "BICYCLE"}

# Unpickling nuPlan objects imports matplotlib through NAVSIM visualization config.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/navsim-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene-path", type=Path, help="Path to a synthetic scene pickle.")
    source.add_argument("--token", help="Synthetic scene token, resolved below --scene-dir.")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=DEFAULT_SCENE_DIR,
        help=f"Synthetic scene directory used with --token (default: {DEFAULT_SCENE_DIR}).",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        help="Only print these types, for example: vehicle pedestrian bicycle.",
    )
    parser.add_argument(
        "--max-step-distance",
        type=float,
        default=DEFAULT_MAX_STEP_DISTANCE_M,
        help="Warn when one track moves farther than this between adjacent stored frames.",
    )
    return parser.parse_args()


def _normalize_type_name(value: str) -> str:
    normalized = value.strip().upper()
    return TYPE_ALIASES.get(normalized.lower(), normalized)


def _resolve_scene_path(args: argparse.Namespace) -> Path:
    path = args.scene_path if args.scene_path is not None else args.scene_dir / f"{args.token}.pkl"
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic scene pickle does not exist: {path}")
    return path


def _load_scene_data(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        scene_data = pickle.load(file)
    if not isinstance(scene_data, dict):
        raise TypeError(f"Expected a scene dictionary in {path}, got {type(scene_data).__name__}")
    return scene_data


def _objects(frame: Any) -> list[Any]:
    return list(frame.tracked_objects.tracked_objects)


def _object_type_name(tracked_object: Any) -> str:
    return tracked_object.tracked_object_type.name


def _timestamp_us(tracked_object: Any) -> int:
    return int(tracked_object.metadata.timestamp_us)


def _format_vector(values: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(value):.3f}" for value in values) + ")"


def _print_scene_header(path: Path, scene_data: dict[str, Any]) -> tuple[int, list[Any]]:
    metadata = scene_data.get("scene_metadata", {})
    frames = scene_data.get("frames", [])
    extended_tracks = scene_data.get("extended_detections_tracks")

    if not frames:
        raise ValueError(f"Scene has no frames: {path}")

    current_timestamp_us = int(frames[-1]["timestamp"])
    print(f"scene_path: {path}")
    print(f"scene_token: {metadata.get('scene_token')}")
    print(f"initial_token: {metadata.get('initial_token')}")
    print(f"log_name: {metadata.get('log_name')}")
    print(f"map_name: {metadata.get('map_name')}")
    print(f"history_frames: {metadata.get('num_history_frames')}")
    print(f"future_frames: {metadata.get('num_future_frames')}")
    print(f"stored_frames: {len(frames)}")
    print(f"current_timestamp_us: {current_timestamp_us}")
    print(
        "extended_frames: "
        + ("None" if extended_tracks is None else str(len(extended_tracks)))
    )

    if extended_tracks is None:
        print("\nWARNING: extended_detections_tracks is None.")
        return current_timestamp_us, []
    if not extended_tracks:
        print("\nWARNING: extended_detections_tracks is empty.")
        return current_timestamp_us, []
    return current_timestamp_us, list(extended_tracks)


def _print_frames(
    extended_tracks: list[Any],
    current_timestamp_us: int,
    selected_types: set[str] | None,
    max_step_distance_m: float,
) -> None:
    track_states: dict[str, list[tuple[float, float, float, str]]] = defaultdict(list)
    frame_times_s: list[float] = []

    for frame_index, frame in enumerate(extended_tracks, start=1):
        all_objects = _objects(frame)
        timestamps = sorted({_timestamp_us(obj) for obj in all_objects})
        timestamp_us = timestamps[0] if timestamps else None
        relative_time_s = (
            (timestamp_us - current_timestamp_us) * 1e-6 if timestamp_us is not None else math.nan
        )
        frame_times_s.append(relative_time_s)

        type_counts = Counter(_object_type_name(obj) for obj in all_objects)
        visible_objects = [
            obj
            for obj in all_objects
            if selected_types is None or _object_type_name(obj) in selected_types
        ]

        print()
        print(
            f"[frame {frame_index:02d}] t={relative_time_s:.3f}s "
            f"timestamp_us={timestamp_us} objects={len(all_objects)} "
            f"types={dict(sorted(type_counts.items()))}"
        )
        if len(timestamps) > 1:
            print(f"  WARNING: frame contains multiple timestamps: {timestamps}")

        for obj in sorted(
            visible_objects,
            key=lambda item: (_object_type_name(item), str(item.track_token)),
        ):
            type_name = _object_type_name(obj)
            center = obj.center
            velocity = getattr(obj, "velocity", None)
            velocity_text = (
                _format_vector((velocity.x, velocity.y)) if velocity is not None else "N/A"
            )
            print(
                f"  token={obj.track_token} type={type_name:<14} "
                f"position={_format_vector((center.x, center.y))} "
                f"heading={center.heading:.3f} velocity={velocity_text} "
                f"size={_format_vector((obj.box.length, obj.box.width, obj.box.height))}"
            )
            track_states[str(obj.track_token)].append(
                (relative_time_s, float(center.x), float(center.y), type_name)
            )

    print("\nTrack summary")
    print("-------------")
    for track_token, states in sorted(track_states.items()):
        first_time, first_x, first_y, type_name = states[0]
        last_time, last_x, last_y, _ = states[-1]
        cumulative_distance = 0.0
        max_step_distance = 0.0
        for previous, current in zip(states, states[1:]):
            step_distance = math.hypot(current[1] - previous[1], current[2] - previous[2])
            cumulative_distance += step_distance
            max_step_distance = max(max_step_distance, step_distance)
        warning = (
            f" WARNING: max_step={max_step_distance:.3f}m"
            if max_step_distance > max_step_distance_m
            else ""
        )
        print(
            f"token={track_token} type={type_name:<14} frames={len(states):02d} "
            f"time=[{first_time:.3f}, {last_time:.3f}]s "
            f"start={_format_vector((first_x, first_y))} "
            f"end={_format_vector((last_x, last_y))} "
            f"distance={cumulative_distance:.3f}m{warning}"
        )

    print("\nValidation")
    print("----------")
    finite_times = [time_s for time_s in frame_times_s if math.isfinite(time_s)]
    if finite_times:
        intervals = [
            current - previous for previous, current in zip(finite_times, finite_times[1:])
        ]
        irregular = [
            interval
            for interval in intervals
            if not math.isclose(interval, DEFAULT_EXPECTED_INTERVAL_S, abs_tol=0.02)
        ]
        print(f"relative_times_s: {[round(value, 3) for value in finite_times]}")
        print(f"intervals_s: {[round(value, 3) for value in intervals]}")
        if irregular:
            print(f"WARNING: expected approximately 0.5s intervals, got {irregular}")
        if finite_times[-1] + 0.02 < DEFAULT_EXPECTED_HORIZON_S:
            print(
                f"WARNING: future horizon is {finite_times[-1]:.3f}s, "
                f"shorter than {DEFAULT_EXPECTED_HORIZON_S:.1f}s."
            )
        else:
            print(f"future_horizon_s: {finite_times[-1]:.3f}")
    else:
        print("WARNING: no object timestamps were available to determine the future horizon.")


def _print_current_frame_continuity(
    scene_data: dict[str, Any],
    extended_tracks: list[Any],
    selected_types: set[str] | None,
) -> None:
    annotations = scene_data["frames"][-1]["annotations"]
    current_names = annotations["names"]
    current_tokens = annotations["track_tokens"]
    future_frames_by_token: dict[str, list[int]] = defaultdict(list)
    future_ego_objects = []

    for frame_index, frame in enumerate(extended_tracks, start=1):
        for obj in _objects(frame):
            future_frames_by_token[str(obj.track_token)].append(frame_index)
            if _object_type_name(obj) == "EGO" or str(obj.track_token) == "ego":
                future_ego_objects.append((frame_index, obj))

    continuity_types = selected_types or DEFAULT_CONTINUITY_TYPES
    current_objects = [
        (_normalize_type_name(name), str(token))
        for name, token in zip(current_names, current_tokens)
        if _normalize_type_name(name) in continuity_types
    ]

    print("\nCurrent-frame continuity")
    print("------------------------")
    if not current_objects:
        print(f"No current-frame objects matched types {sorted(continuity_types)}.")
    else:
        counts_by_type: dict[str, Counter[str]] = defaultdict(Counter)
        for type_name, track_token in sorted(current_objects):
            future_frames = future_frames_by_token.get(track_token, [])
            counts_by_type[type_name]["current"] += 1
            if future_frames:
                counts_by_type[type_name]["seen_later"] += 1
            if len(future_frames) == len(extended_tracks):
                counts_by_type[type_name]["all_extended_frames"] += 1
            print(
                f"token={track_token} type={type_name:<14} "
                f"future_frames={future_frames or 'NONE'}"
            )

        print("\nContinuity summary")
        for type_name in sorted(counts_by_type):
            counts = counts_by_type[type_name]
            current = counts["current"]
            print(
                f"type={type_name:<14} current={current} "
                f"seen_later={counts['seen_later']}/{current} "
                f"in_every_extended_frame={counts['all_extended_frames']}/{current}"
            )

    print("\nEgo check")
    print("---------")
    print("The current ego is stored in frame['ego_status'], not in frame annotations.")
    if future_ego_objects:
        print(
            "WARNING: found EGO/track_token='ego' objects in extended detections at frames "
            f"{[frame_index for frame_index, _ in future_ego_objects]}"
        )
    else:
        print("extended_detections_tracks contains no EGO object and no track_token='ego'.")


def main() -> None:
    args = _parse_args()
    scene_path = _resolve_scene_path(args)
    scene_data = _load_scene_data(scene_path)
    selected_types = (
        {_normalize_type_name(value) for value in args.types} if args.types else None
    )

    current_timestamp_us, extended_tracks = _print_scene_header(scene_path, scene_data)
    if not extended_tracks:
        return

    if selected_types is not None:
        print(f"type_filter: {sorted(selected_types)}")
    _print_current_frame_continuity(scene_data, extended_tracks, selected_types)
    _print_frames(
        extended_tracks,
        current_timestamp_us,
        selected_types,
        args.max_step_distance,
    )


if __name__ == "__main__":
    main()
