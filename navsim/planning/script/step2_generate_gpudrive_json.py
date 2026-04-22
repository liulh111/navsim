"""
Step 2: Generate a full GPUDrive JSON from scene metadata (run in ScenarioMax .venv).

For each metadata pickle emitted by Step 1, this script

  1. Reconstructs the nuPlan map API for the scene's map_name
  2. Calls ScenarioMax's official `extract_static_map_elements` and
     `convert_map_features` to get GPUDrive-style road features
  3. Applies the same area-based polyline simplification that GPUDrive's
     C++ loader uses (so we never exceed kMaxRoadEntityCount and the
     geometry is bit-identical to the standard pipeline)
  4. Writes a full GPUDrive scenario JSON containing:
        - the real ego at centered origin (91 frames) with actual
          velocity/goal
        - all partners in natural annotation order (capped at 63) with
          centered positions, headings and velocities

Output matches the convention used by the standard ScenarioMax → nuPlan →
GPUDrive pipeline: `sdc_track_index=0`, `tracks_to_predict=[]`.

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

EPISODE_LEN = 91            # must match madrona_gpudrive.episodeLen


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


def _build_ego_object(metadata):
    h = float(metadata["ego_pose"][2])
    vx_local, vy_local = metadata["ego_velocity_local"]
    vx_c, vy_c = _rotate((vx_local, vy_local), h)
    gx_local, gy_local = metadata["goal_local"]
    gx_c, gy_c = _rotate((gx_local, gy_local), h)
    length, width, height = metadata["ego_size"]

    return {
        "position": [{"x": 0.0, "y": 0.0, "z": 0.0}] * EPISODE_LEN,
        "width": float(width),
        "length": float(length),
        "height": float(height),
        "heading": [h] * EPISODE_LEN,
        "velocity": [{"x": vx_c, "y": vy_c}] * EPISODE_LEN,
        "valid": [True] * EPISODE_LEN,
        "goalPosition": {"x": gx_c, "y": gy_c, "z": 0.0},
        "type": "vehicle",
        "id": 0,
        "mark_as_expert": False,
    }


def _build_partner_object(idx, partner, ego_heading):
    px_c, py_c = _rotate((partner["rel_x"], partner["rel_y"]), ego_heading)
    vx_c, vy_c = _rotate((partner["vel_x"], partner["vel_y"]), ego_heading)
    head_c = float(partner["heading"]) + ego_heading
    return {
        "position": [{"x": px_c, "y": py_c, "z": 0.0}] * EPISODE_LEN,
        "width": float(partner["width"]),
        "length": float(partner["length"]),
        "height": float(partner.get("height", 1.5) or 1.5),
        "heading": [head_c] * EPISODE_LEN,
        "velocity": [{"x": vx_c, "y": vy_c}] * EPISODE_LEN,
        "valid": [True] * EPISODE_LEN,
        "goalPosition": {"x": px_c, "y": py_c, "z": 0.0},
        "type": "vehicle",
        "id": int(idx),
        "mark_as_expert": False,
    }


def build_gpudrive_json_full(metadata, road_features):
    """Assemble a GPUDrive scenario dict with real ego + partners + roads."""
    ego_heading = float(metadata["ego_pose"][2])

    objects = [_build_ego_object(metadata)]

    partners = metadata.get("partners_local", [])
    if len(partners) > MAX_PARTNER_COUNT:
        logger.warning(
            f"  Scene has {len(partners)} partners > cap ({MAX_PARTNER_COUNT}); "
            f"taking first {MAX_PARTNER_COUNT} in annotation order."
        )
        partners = partners[:MAX_PARTNER_COUNT]

    for i, partner in enumerate(partners):
        objects.append(_build_partner_object(i + 1, partner, ego_heading))

    return {
        "name": f"tfrecord-{metadata['token']}.json",
        "scenario_id": metadata["token"],
        "objects": objects,
        "roads": road_features,
        "tl_states": {},
        "metadata": {
            "sdc_track_index": 0,
            "objects_of_interest": [],
            "tracks_to_predict": [],
        },
    }


# ── Pipeline ──

def process_one(metadata_path):
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    token = metadata["token"]
    stage = metadata.get("stage", "unknown")
    map_name = metadata["map_name"]
    center = [float(metadata["ego_pose"][0]), float(metadata["ego_pose"][1])]

    logger.info("=" * 72)
    logger.info(f"Token:      {token}  (stage={stage})")
    logger.info(f"Map:        {map_name}")
    logger.info(f"Center:     {center}")
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

    road_features, total_segments = simplify_road_features(
        road_features, POLYLINE_REDUCTION_THRESHOLD
    )

    # Fallback: if still above the cap, drop the furthest roads
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

    scenario_dict = build_gpudrive_json_full(metadata, road_features)
    scenario_dict = convert_numpy(scenario_dict)

    output_dir = os.path.join(OUTPUT_BASE, token, "gpudrive_json")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tfrecord-{token}.json")
    with open(output_path, "w") as f:
        json.dump(scenario_dict, f)

    logger.info(f"Saved GPUDrive JSON: {output_path}")
    logger.info(f"  Objects: {len(scenario_dict['objects'])} "
                f"(ego + {len(scenario_dict['objects']) - 1} partners)")


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
        process_one(p)


if __name__ == "__main__":
    main()
