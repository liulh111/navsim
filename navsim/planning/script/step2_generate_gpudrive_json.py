"""
Step 2: Generate a ScenarioMax-like GPUDrive JSON from scene metadata
(run in ScenarioMax .venv).

For each metadata pickle emitted by Step 1, this script

  1. Reconstructs the nuPlan map API for the scene's map_name
  2. Calls ScenarioMax's official `extract_static_map_elements` and
     `convert_map_features` to get GPUDrive-style road features
  3. Writes the road geometry without pre-simplification by default, matching
     ScenarioMax's JSON export. GPUDrive can still simplify it when loading.
  4. Writes ScenarioMax-style objects for the current frame:
        - ego and current-frame annotations only
        - fields/metadata matching `unified_to_gpudrive.convert_to_json`
        - state repeated across the 90 ScenarioMax nuPlan iterations

This intentionally does not recreate agents that only appear in history/future
nuPlan frames; navsim's stage-1/stage-2 inputs expose the current frame that
the planner sees.

Usage:
    /data/llh/navsim_workspace/ScenarioMax/.venv/bin/python \\
        /data/llh/navsim_workspace/navsim/navsim/planning/script/step2_generate_gpudrive_json.py \\
        --metadata_path <stage1_pkl> --metadata_path <stage2_pkl>

    # or, auto-discover every scene_metadata.pkl under OUTPUT_BASE:
    /data/llh/navsim_workspace/ScenarioMax/.venv/bin/python \\
        /data/llh/navsim_workspace/navsim/navsim/planning/script/step2_generate_gpudrive_json.py --auto
"""
import argparse
import glob
import json
import logging
import math
import os
import pickle
import sys

import numpy as np

# ScenarioMax imports
sys.path.insert(0, "/data/llh/navsim_workspace/ScenarioMax")

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from scenariomax.raw_to_unified.datasets.nuplan.extractor import extract_static_map_elements
from scenariomax.unified_to_gpudrive.converter.roadgraph import convert_map_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NUPLAN_MAPS_ROOT = os.environ.get(
    "NUPLAN_MAPS_ROOT", "/data/llh/navsim_workspace/dataset/maps"
)
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"

OUTPUT_BASE = "/data/llh/navsim_workspace/exp/pipeline_output"

# Must match gpudrive/gpudrive/env/config.py polyline_reduction_threshold
POLYLINE_REDUCTION_THRESHOLD = 0.1
# Must match gpudrive/src/consts.hpp kMaxRoadEntityCount / kMaxAgentCount
MAX_ROAD_ENTITY_COUNT = 10000
MAX_AGENT_COUNT = 64        # ego + 63 partners
MAX_PARTNER_COUNT = MAX_AGENT_COUNT - 1

EPISODE_LEN = 90            # ScenarioMax NuPlan 9.0s / 0.1s JSON length

# NAVSIM annotations are ego-rear-axle local. ScenarioMax's nuPlan extractor
# uses EgoState.waypoint, which is the ego box center, as the GPUDrive object
# position. Convert partner/road centering through the same ego-center frame.
EGO_REAR_AXLE_TO_CENTER = float(get_pacifica_parameters().rear_axle_to_center)

GPUDRIVE_AGENT_TYPES = {
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "bicycle": "cyclist",
    "cyclist": "cyclist",
}

# ── Geometry simplification (matches GPUDrive C++ json_serialization.hpp) ──

def simplify_geometry(points, threshold):
    if len(points) < 10:
        return points

    skip = [False] * len(points)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(points):
            if skip[i]:
                i += 1
                continue
            j = i + 1
            while j < len(points) and skip[j]:
                j += 1
            if j >= len(points):
                break
            k = j + 1
            while k < len(points) and skip[k]:
                k += 1
            if k >= len(points):
                break
            p1, p2, p3 = points[i], points[j], points[k]
            area = 0.5 * abs(
                (p1["x"] - p3["x"]) * (p2["y"] - p1["y"])
                - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"])
            )
            if area < threshold:
                skip[j] = True
                changed = True
            i = j

    return [p for p, s in zip(points, skip) if not s]


def simplify_road_features(road_features, threshold):
    total_before = sum(len(r["geometry"]) for r in road_features)
    for road in road_features:
        road["geometry"] = simplify_geometry(road["geometry"], threshold)
    total_after = sum(len(r["geometry"]) for r in road_features)
    total_segments = sum(max(len(r["geometry"]) - 1, 0) for r in road_features)
    logger.info(
        f"  Simplification: {total_before} pts -> {total_after} pts "
        f"({total_segments} segments)"
    )
    return road_features, total_segments


def convert_numpy(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj


# ── Objects (ego + partners) in the GPUDrive centered-global frame ──

def _rotate(vec_xy, heading):
    """Rotate a 2D vector by `heading` radians (ego-local -> centered-global)."""
    c, s = math.cos(heading), math.sin(heading)
    x, y = float(vec_xy[0]), float(vec_xy[1])
    return c * x - s * y, s * x + c * y


def _ego_center_global(metadata):
    """Return the ego center in global map coordinates."""
    x, y, h = metadata["ego_pose"]
    dx, dy = _rotate((EGO_REAR_AXLE_TO_CENTER, 0.0), float(h))
    return [float(x) + dx, float(y) + dy]


def _partner_type(partner):
    """Map NAVSIM annotation names to GPUDrive-supported dynamic agent types."""
    raw_name = str(partner.get("name", "")).strip().lower()
    return GPUDRIVE_AGENT_TYPES.get(raw_name)


def _wrap_to_pi(angle):
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _as_object_id(object_id, fallback_index):
    object_id = str(object_id)
    return int(object_id) if object_id.isdigit() else int(fallback_index)


def _repeat_position(x, y, z=0.0):
    return [{"x": float(x), "y": float(y), "z": float(z)} for _ in range(EPISODE_LEN)]


def _repeat_velocity(vx, vy):
    return [{"x": float(vx), "y": float(vy)} for _ in range(EPISODE_LEN)]


def _scenario_name(metadata):
    token = metadata["token"]
    return f"nuPlan_navsim_current_frame_{token}.json"


def _scenario_type(metadata):
    token = metadata["token"]
    return f"manual_navsim_current_frame+{token}"


def _partner_object_id(partner, fallback_index):
    for key in ("track_token", "instance_token", "annotation_index"):
        value = partner.get(key)
        if value not in (None, ""):
            return str(value)
    return str(fallback_index)


def _ordered_entries(metadata, order_strategy):
    """Return [(object_id, partner_or_none), ...] including ego.

    ScenarioMax's nuPlan extractor builds dynamic_agents from a set of
    track_tokens plus "ego", so its object order is not annotation order and
    the SDC is not necessarily object 0. `annotation` is deterministic and
    keeps ego first. The optional `scenariomax_set` mode mirrors ScenarioMax's
    set construction but inherits Python hash-seed dependent ordering.
    """
    partners = list(metadata.get("partners_local", []))

    if order_strategy == "annotation":
        return [("ego", None)] + [
            (_partner_object_id(partner, i), partner)
            for i, partner in enumerate(partners)
        ]

    partner_by_id = {}
    all_ids = {"ego"}
    for i, partner in enumerate(partners):
        object_id = _partner_object_id(partner, i)
        all_ids.add(object_id)
        partner_by_id.setdefault(object_id, partner)

    entries = []
    for object_id in all_ids:
        if object_id == "ego":
            entries.append((object_id, None))
        else:
            entries.append((object_id, partner_by_id[object_id]))
    return entries


def _build_ego_object(index, metadata):
    h = float(metadata["ego_pose"][2])
    vx_local, vy_local = metadata["ego_velocity_local"]
    vx_c, vy_c = _rotate((vx_local, vy_local), h)
    gx_local, gy_local = metadata["goal_local"]
    gx_c, gy_c = _rotate((gx_local, gy_local), h)
    length, width, height = metadata["ego_size"]

    return {
        "id": int(index),
        "type": "vehicle",
        "position": _repeat_position(0.0, 0.0, 0.0),
        "width": float(width),
        "length": float(length),
        "height": float(height),
        "heading": [_wrap_to_pi(h)] * EPISODE_LEN,
        "velocity": _repeat_velocity(vx_c, vy_c),
        "valid": [True] * EPISODE_LEN,
        "goalPosition": {"x": gx_c, "y": gy_c, "z": 0.0},
        "is_sdc": True,
        "mark_as_expert": False,
        "total_distance_traveled": 0.0,
    }


def _build_partner_object(index, object_id, partner, ego_heading):
    # NAVSIM boxes are relative to ego rear axle, while ScenarioMax/GPUDrive
    # observes agents relative to the ego box center.
    px_local_center = float(partner["rel_x"]) - EGO_REAR_AXLE_TO_CENTER
    py_local_center = float(partner["rel_y"])
    px_c, py_c = _rotate((px_local_center, py_local_center), ego_heading)
    vx_c, vy_c = _rotate((partner["vel_x"], partner["vel_y"]), ego_heading)
    head_c = _wrap_to_pi(float(partner["heading"]) + ego_heading)
    return {
        "id": _as_object_id(object_id, index),
        "type": _partner_type(partner),
        "position": _repeat_position(px_c, py_c, 0.0),
        "width": float(partner["width"]),
        "length": float(partner["length"]),
        "height": float(partner.get("height", 1.5) or 1.5),
        "heading": [head_c] * EPISODE_LEN,
        "velocity": _repeat_velocity(vx_c, vy_c),
        "valid": [True] * EPISODE_LEN,
        "goalPosition": {"x": px_c, "y": py_c, "z": 0.0},
        "is_sdc": False,
        "mark_as_expert": False,
        "total_distance_traveled": 0.0,
    }


def build_gpudrive_json_full(metadata, road_features, order_strategy):
    """Assemble a ScenarioMax-like GPUDrive scenario dict."""
    ego_heading = float(metadata["ego_pose"][2])

    entries = _ordered_entries(metadata, order_strategy)
    if len(entries) > MAX_AGENT_COUNT:
        logger.warning(
            f"  Scene has {len(entries)} objects > visible agent cap "
            f"({MAX_AGENT_COUNT}); keeping JSON order and letting GPUDrive "
            "apply its agent cap after type filtering."
        )

    objects = []
    skipped_by_type = {}
    sdc_track_index = 0
    for object_id, partner in entries:
        output_index = len(objects)
        if partner is None:
            sdc_track_index = output_index
            objects.append(_build_ego_object(output_index, metadata))
            continue

        if _partner_type(partner) is None:
            raw_name = str(partner.get("name", "")).strip().lower() or "<missing>"
            skipped_by_type[raw_name] = skipped_by_type.get(raw_name, 0) + 1
            continue
        objects.append(_build_partner_object(output_index, object_id, partner, ego_heading))

    if skipped_by_type:
        logger.info(
            "  Skipped unsupported partner types: %s",
            ", ".join(
                f"{name}={count}" for name, count in sorted(skipped_by_type.items())
            ),
        )

    return {
        "name": _scenario_name(metadata),
        "scenario_id": metadata["token"],
        "objects": objects,
        "roads": road_features,
        "tl_states": {},
        "metadata": {
            "sdc_track_index": int(sdc_track_index),
            "log_name": metadata.get("log_name", ""),
            "initial_lidar_timestamp": int(metadata.get("timestamp", 0) or 0),
            "map_name": metadata.get("map_name", ""),
            "objects_of_interest": [],
            "tracks_to_predict": [],
            "average_distance_traveled": 0.0,
            "scenario_type": _scenario_type(metadata),
        },
    }


# ── Pipeline ──

def process_one(metadata_path, args):
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    token = metadata["token"]
    stage = metadata.get("stage", "unknown")
    map_name = metadata["map_name"]
    center = _ego_center_global(metadata)

    logger.info("=" * 72)
    logger.info(f"Token:      {token}  (stage={stage})")
    logger.info(f"Map:        {map_name}")
    logger.info(f"Center:     {center}  (ego box center)")
    logger.info(f"Heading:    {metadata['ego_pose'][2]:.4f} rad")
    logger.info(f"#partners:  {len(metadata.get('partners_local', []))}")

    logger.info(f"Loading map from {NUPLAN_MAPS_ROOT} ...")
    map_api = get_maps_api(NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, map_name)

    logger.info("Extracting static map elements via ScenarioMax ...")
    static_map_elements = extract_static_map_elements(map_api, center)
    logger.info(f"  Extracted {len(static_map_elements)} map elements")

    logger.info("Converting to GPUDrive road format ...")
    road_features, _edge_segments = convert_map_features(static_map_elements)
    if road_features is None:
        logger.error("3D structure detected — scenario rejected by convert_map_features")
        return
    logger.info(f"  Road features: {len(road_features)}")

    if args.simplify_roads:
        road_features, total_segments = simplify_road_features(
            road_features, POLYLINE_REDUCTION_THRESHOLD
        )

        # Fallback: if still above the cap, drop the furthest roads.
        if total_segments > MAX_ROAD_ENTITY_COUNT:
            logger.warning(
                f"  Total segments ({total_segments}) > kMaxRoadEntityCount "
                f"({MAX_ROAD_ENTITY_COUNT}); pruning distant roads."
            )
            for r in road_features:
                xs = [p["x"] for p in r["geometry"]]
                ys = [p["y"] for p in r["geometry"]]
                r["_dist"] = (sum(xs) / len(xs)) ** 2 + (sum(ys) / len(ys)) ** 2
            road_features.sort(key=lambda r: r["_dist"])
            kept, seg_count = [], 0
            for r in road_features:
                segs = max(len(r["geometry"]) - 1, 0)
                if seg_count + segs > MAX_ROAD_ENTITY_COUNT:
                    break
                kept.append(r)
                seg_count += segs
            for r in kept:
                del r["_dist"]
            road_features = kept
            logger.info(f"  After pruning: {len(road_features)} roads, {seg_count} segments")
    else:
        total_points = sum(len(r["geometry"]) for r in road_features)
        logger.info(
            "  Keeping raw ScenarioMax road geometry: %d roads, %d points",
            len(road_features),
            total_points,
        )

    scenario_dict = build_gpudrive_json_full(
        metadata,
        road_features,
        order_strategy=args.object_order,
    )
    scenario_dict = convert_numpy(scenario_dict)

    output_dir = os.path.join(OUTPUT_BASE, token, "gpudrive_json")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tfrecord-{token}.json")
    with open(output_path, "w") as f:
        json.dump(scenario_dict, f)

    logger.info(f"Saved GPUDrive JSON: {output_path}")
    logger.info(f"  Objects: {len(scenario_dict['objects'])} "
                f"(sdc_index={scenario_dict['metadata']['sdc_track_index']})")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate full GPUDrive JSON")
    parser.add_argument(
        "--metadata_path", action="append", default=[],
        help="Path to scene_metadata.pkl from Step 1 (repeatable).",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help=f"Auto-discover all scene_metadata.pkl under {OUTPUT_BASE}.",
    )
    parser.add_argument(
        "--object_order",
        choices=["scenariomax_set", "annotation"],
        default="annotation",
        help=(
            "Object ordering strategy. `annotation` keeps ego first and "
            "partners in navsim annotation order (default, reproducible). "
            "`scenariomax_set` mirrors the nuPlan ScenarioMax extractor's "
            "set(track_token)+ego behavior but is hash-seed dependent."
        ),
    )
    parser.add_argument(
        "--simplify_roads",
        action="store_true",
        help=(
            "Apply the old Python road simplification before writing JSON. "
            "Default is false to match ScenarioMax JSON export more closely."
        ),
    )
    args = parser.parse_args()

    paths = list(args.metadata_path)
    if args.auto:
        paths.extend(sorted(glob.glob(
            os.path.join(OUTPUT_BASE, "*", "scene_metadata.pkl")
        )))

    # de-dupe while preserving order
    seen, ordered = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    if not ordered:
        logger.error("No metadata paths given. Use --metadata_path or --auto.")
        sys.exit(1)

    for p in ordered:
        process_one(p, args)


if __name__ == "__main__":
    main()
