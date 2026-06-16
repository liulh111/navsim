#!/usr/bin/env python
"""Validate exported GPUDrive JSON files against reference exports.

The validator intentionally compares scene semantics instead of doing a raw JSON
diff. ScenarioMax exports and NAVSIM exports may assign different object ids and
may include different invalid tracks, while still agreeing on the ego frame,
map geometry, and current-frame agent geometry.

GPUDrive scene objects are expected to use the array-field representation:
``position``, ``heading``, ``velocity``, and ``valid``. A ``states`` list is not
accepted as a valid GPUDrive export by this validator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"([0-9a-fA-F]{16,})$")
ERR_VAL = -1e4


@dataclass
class Thresholds:
    ego_position: float = 1e-3
    ego_heading: float = 1e-4
    ego_velocity: float = 1e-3
    agent_position: float = 0.1
    road_point: float = 1e-3


@dataclass
class ObjectView:
    object_id: str
    object_type: str
    is_sdc: bool
    length: float
    width: float
    height: float
    position: np.ndarray
    heading: np.ndarray
    velocity: np.ndarray
    valid: np.ndarray


@dataclass
class SceneView:
    path: Path
    token: str
    objects: list[ObjectView]
    roads: list[dict[str, Any]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-dir", type=Path, required=True, help="Directory containing JSON files to validate.")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Directory containing reference JSON files.")
    parser.add_argument(
        "--mode",
        choices=("current_frame", "stage_one_scenariomax"),
        required=True,
        help="Comparison mode. current_frame compares t=0; stage_one_scenariomax also compares ego future.",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--report-path", type=Path, default=None, help="Optional CSV report path.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of worst scenes to print.")
    parser.add_argument("--ego-position-threshold", type=float, default=Thresholds.ego_position)
    parser.add_argument("--ego-heading-threshold", type=float, default=Thresholds.ego_heading)
    parser.add_argument("--ego-velocity-threshold", type=float, default=Thresholds.ego_velocity)
    parser.add_argument("--agent-position-threshold", type=float, default=Thresholds.agent_position)
    parser.add_argument("--road-point-threshold", type=float, default=Thresholds.road_point)
    parser.add_argument(
        "--allow-missing-reference",
        action="store_true",
        help="Skip new JSON files without a matching reference instead of failing them.",
    )
    return parser.parse_args()


def _token_from_path(path: Path) -> str:
    match = TOKEN_RE.search(path.stem)
    if match:
        return match.group(1).lower()
    return path.stem.lower()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _is_likely_scene_json(path: Path) -> bool:
    name = path.name
    if not name.endswith(".json"):
        return False
    if name.endswith(".idm_routes.json"):
        return False
    if name in {"manifest.json", "summary.json"} or "manifest" in name:
        return False
    return True


def _scene_files(scene_dir: Path, max_scenes: int | None) -> dict[str, Path]:
    files: list[Path] = []
    for path in scene_dir.iterdir():
        if not path.is_file() or not _is_likely_scene_json(path):
            continue
        files.append(path)
        if max_scenes is not None and len(files) >= max_scenes:
            break
    return {_token_from_path(path): path for path in sorted(files)}


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _point3(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [_as_float(value.get("x")), _as_float(value.get("y")), _as_float(value.get("z", 0.0), 0.0)]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        z = value[2] if len(value) > 2 else 0.0
        return [_as_float(value[0]), _as_float(value[1]), _as_float(z, 0.0)]
    return [math.nan, math.nan, math.nan]


def _point2(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [_as_float(value.get("x")), _as_float(value.get("y"))]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [_as_float(value[0]), _as_float(value[1])]
    return [math.nan, math.nan]


def _normalize_object(obj: dict[str, Any]) -> ObjectView:
    if "states" in obj:
        raise ValueError("object uses unsupported states-list format; expected GPUDrive array fields")

    required_fields = ("position", "heading", "velocity", "valid")
    missing_fields = [field for field in required_fields if field not in obj]
    if missing_fields:
        raise ValueError(f"object {obj.get('id')} missing required fields: {missing_fields}")

    position = np.asarray([_point3(point) for point in obj.get("position", [])], dtype=np.float64)
    heading = np.asarray([_as_float(value) for value in obj.get("heading", [])], dtype=np.float64)
    velocity = np.asarray([_point2(value) for value in obj.get("velocity", [])], dtype=np.float64)
    valid = np.asarray([bool(value) for value in obj.get("valid", [])], dtype=bool)

    n = min(len(position), len(heading), len(velocity), len(valid))
    position = position[:n]
    heading = heading[:n]
    velocity = velocity[:n]
    valid = valid[:n]

    invalid_by_sentinel = np.isclose(position[:, 0], ERR_VAL) | np.isclose(position[:, 1], ERR_VAL) if n else np.zeros(0)
    valid = valid & ~invalid_by_sentinel

    return ObjectView(
        object_id=str(obj.get("id", "")),
        object_type=str(obj.get("type", "unknown")).lower(),
        is_sdc=bool(obj.get("is_sdc", False)),
        length=_as_float(obj.get("length")),
        width=_as_float(obj.get("width")),
        height=_as_float(obj.get("height")),
        position=position,
        heading=heading,
        velocity=velocity,
        valid=valid,
    )


def _load_scene(path: Path) -> SceneView:
    data = _load_json(path)
    objects = [_normalize_object(obj) for obj in data.get("objects", []) if isinstance(obj, dict)]
    return SceneView(path=path, token=_token_from_path(path), objects=objects, roads=list(data.get("roads", [])))


def _ego(scene: SceneView) -> ObjectView | None:
    egos = [obj for obj in scene.objects if obj.is_sdc]
    return egos[0] if egos else None


def _angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def _valid_indices(a: ObjectView, b: ObjectView, max_steps: int | None = None) -> np.ndarray:
    n = min(len(a.valid), len(b.valid))
    if max_steps is not None:
        n = min(n, max_steps)
    if n <= 0:
        return np.asarray([], dtype=int)
    idx = np.arange(n)
    return idx[a.valid[:n] & b.valid[:n]]


def _object_errors(new: ObjectView, ref: ObjectView, max_steps: int | None) -> dict[str, float]:
    idx = _valid_indices(new, ref, max_steps=max_steps)
    if len(idx) == 0:
        return {"pos": math.inf, "heading": math.inf, "velocity": math.inf, "num_steps": 0}
    pos = np.linalg.norm(new.position[idx, :2] - ref.position[idx, :2], axis=1)
    heading = np.abs(_angle_diff(new.heading[idx], ref.heading[idx]))
    velocity = np.linalg.norm(new.velocity[idx] - ref.velocity[idx], axis=1)
    return {
        "pos": float(np.max(pos)),
        "heading": float(np.max(heading)),
        "velocity": float(np.max(velocity)),
        "num_steps": int(len(idx)),
    }


def _objects_at_t0(scene: SceneView, *, exclude_ego: bool = True) -> list[ObjectView]:
    objects = []
    for obj in scene.objects:
        if exclude_ego and obj.is_sdc:
            continue
        if len(obj.valid) > 0 and bool(obj.valid[0]):
            objects.append(obj)
    return objects


def _match_agents(
    new_scene: SceneView,
    ref_scene: SceneView,
    threshold: float,
) -> tuple[list[tuple[ObjectView, ObjectView, float]], list[ObjectView], list[ObjectView]]:
    ref_remaining = _objects_at_t0(ref_scene, exclude_ego=True)
    matches: list[tuple[ObjectView, ObjectView, float]] = []
    unmatched_new: list[ObjectView] = []

    for new_obj in _objects_at_t0(new_scene, exclude_ego=True):
        best_index = None
        best_distance = math.inf
        for index, ref_obj in enumerate(ref_remaining):
            if ref_obj.object_type != new_obj.object_type:
                continue
            distance = float(np.linalg.norm(new_obj.position[0, :2] - ref_obj.position[0, :2]))
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None or best_distance > threshold:
            unmatched_new.append(new_obj)
            continue
        matches.append((new_obj, ref_remaining.pop(best_index), best_distance))

    return matches, unmatched_new, ref_remaining


def _road_key(road: dict[str, Any]) -> tuple[str, str, str]:
    return (str(road.get("type", "")), str(road.get("map_element_id", "")), str(road.get("id", "")))


def _road_points(road: dict[str, Any]) -> np.ndarray:
    return np.asarray([_point3(point) for point in road.get("geometry", [])], dtype=np.float64)


def _compare_roads(new_scene: SceneView, ref_scene: SceneView) -> dict[str, float | int]:
    new_by_key = {_road_key(road): road for road in new_scene.roads}
    ref_by_key = {_road_key(road): road for road in ref_scene.roads}
    common_keys = sorted(set(new_by_key) & set(ref_by_key))
    max_point_error = 0.0
    compared = 0
    for key in common_keys:
        new_points = _road_points(new_by_key[key])
        ref_points = _road_points(ref_by_key[key])
        n = min(len(new_points), len(ref_points))
        if n == 0:
            continue
        error = np.linalg.norm(new_points[:n, :2] - ref_points[:n, :2], axis=1)
        max_point_error = max(max_point_error, float(np.max(error)))
        compared += 1
    return {
        "new_roads": len(new_scene.roads),
        "ref_roads": len(ref_scene.roads),
        "matched_roads": compared,
        "road_max_point_error": max_point_error,
    }


def _count_by_type(objects: list[ObjectView]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in objects:
        counts[obj.object_type] = counts.get(obj.object_type, 0) + 1
    return counts


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _validate_pair(
    token: str,
    new_scene: SceneView,
    ref_scene: SceneView,
    mode: str,
    thresholds: Thresholds,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "token": token,
        "new_path": str(new_scene.path),
        "reference_path": str(ref_scene.path),
        "status": "ok",
        "errors": "",
    }
    errors: list[str] = []

    new_ego = _ego(new_scene)
    ref_ego = _ego(ref_scene)
    if new_ego is None or ref_ego is None:
        errors.append("missing_ego")
        row.update({"status": "fail", "errors": ";".join(errors)})
        return row

    max_steps = 41 if mode == "stage_one_scenariomax" else 1
    ego_errors = _object_errors(new_ego, ref_ego, max_steps=max_steps)
    row.update(
        {
            "ego_max_position_error": ego_errors["pos"],
            "ego_max_heading_error": ego_errors["heading"],
            "ego_max_velocity_error": ego_errors["velocity"],
            "ego_compared_steps": ego_errors["num_steps"],
        }
    )
    if ego_errors["pos"] > thresholds.ego_position:
        errors.append("ego_position")
    if ego_errors["heading"] > thresholds.ego_heading:
        errors.append("ego_heading")
    if ego_errors["velocity"] > thresholds.ego_velocity:
        errors.append("ego_velocity")

    matches, unmatched_new, unmatched_ref = _match_agents(new_scene, ref_scene, thresholds.agent_position)
    max_agent_error = max((distance for _, _, distance in matches), default=0.0)
    row.update(
        {
            "matched_agents_t0": len(matches),
            "unmatched_new_agents_t0": len(unmatched_new),
            "unmatched_reference_agents_t0": len(unmatched_ref),
            "max_agent_position_error_t0": max_agent_error,
            "new_agent_counts_t0": _format_counts(_count_by_type(_objects_at_t0(new_scene, exclude_ego=True))),
            "reference_agent_counts_t0": _format_counts(_count_by_type(_objects_at_t0(ref_scene, exclude_ego=True))),
        }
    )

    road_summary = _compare_roads(new_scene, ref_scene)
    row.update(road_summary)
    if (
        int(road_summary["new_roads"]) == int(road_summary["ref_roads"])
        and int(road_summary["matched_roads"]) == int(road_summary["new_roads"])
        and float(road_summary["road_max_point_error"]) > thresholds.road_point
    ):
        errors.append("road_geometry")

    row["status"] = "fail" if errors else "ok"
    row["errors"] = ";".join(errors)
    return row


def _write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]], top_k: int) -> None:
    ok = sum(1 for row in rows if row.get("status") == "ok")
    fail = len(rows) - ok
    print(f"Compared scenes: {len(rows)}")
    print(f"OK: {ok}")
    print(f"Failed: {fail}")
    if not rows:
        return

    def max_float(key: str) -> float:
        values = [float(row[key]) for row in rows if key in row and row[key] not in ("", None)]
        return max(values) if values else math.nan

    print(f"Max ego position error: {max_float('ego_max_position_error'):.6g}")
    print(f"Max ego heading error: {max_float('ego_max_heading_error'):.6g}")
    print(f"Max ego velocity error: {max_float('ego_max_velocity_error'):.6g}")
    print(f"Max agent t0 position error: {max_float('max_agent_position_error_t0'):.6g}")
    print(f"Max road point error: {max_float('road_max_point_error'):.6g}")

    failed_rows = [row for row in rows if row.get("status") != "ok"]
    if failed_rows:
        print("\nFailed scenes:")
        for row in failed_rows[:top_k]:
            print(
                f"  {row.get('token')}: {row.get('errors')} "
                f"ego_pos={row.get('ego_max_position_error')} "
                f"agents={row.get('matched_agents_t0')}/"
                f"{row.get('unmatched_new_agents_t0')}/"
                f"{row.get('unmatched_reference_agents_t0')}"
            )

    ranked = sorted(
        rows,
        key=lambda row: float(row.get("ego_max_position_error", -1.0))
        if row.get("ego_max_position_error") not in ("", None)
        else -1.0,
        reverse=True,
    )
    print("\nWorst ego position scenes:")
    for row in ranked[:top_k]:
        print(f"  {row.get('token')}: {row.get('ego_max_position_error')}")


def main() -> int:
    args = _parse_args()
    thresholds = Thresholds(
        ego_position=args.ego_position_threshold,
        ego_heading=args.ego_heading_threshold,
        ego_velocity=args.ego_velocity_threshold,
        agent_position=args.agent_position_threshold,
        road_point=args.road_point_threshold,
    )

    new_files = _scene_files(args.new_dir, args.max_scenes)
    ref_files = _scene_files(args.reference_dir, None)
    if not new_files:
        raise FileNotFoundError(f"No scene JSON files found in {args.new_dir}")
    if not ref_files:
        raise FileNotFoundError(f"No scene JSON files found in {args.reference_dir}")

    rows: list[dict[str, Any]] = []
    for token, new_path in new_files.items():
        ref_path = ref_files.get(token)
        if ref_path is None:
            row = {
                "token": token,
                "new_path": str(new_path),
                "reference_path": "",
                "status": "skip" if args.allow_missing_reference else "fail",
                "errors": "missing_reference",
            }
            rows.append(row)
            continue
        try:
            row = _validate_pair(
                token=token,
                new_scene=_load_scene(new_path),
                ref_scene=_load_scene(ref_path),
                mode=args.mode,
                thresholds=thresholds,
            )
        except Exception as error:  # noqa: BLE001
            row = {
                "token": token,
                "new_path": str(new_path),
                "reference_path": str(ref_path),
                "status": "fail",
                "errors": repr(error),
            }
        rows.append(row)

    if args.report_path is not None:
        _write_report(args.report_path, rows)
        print(f"Wrote report: {args.report_path}")

    _print_summary(rows, args.top_k)
    return 1 if any(row.get("status") == "fail" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
