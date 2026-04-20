"""
Step 2: Generate GPUDrive JSON from scene metadata (run in ScenarioMax .venv).

Reads the metadata pickle from Step 1, reconstructs the nuPlan map API,
calls ScenarioMax's extraction and conversion functions, and writes a
minimal GPUDrive-compatible JSON file.

Usage:
    /data/llh/navsim_workspace/ScenarioMax/.venv/bin/python \
        /data/llh/navsim_workspace/navsim/navsim/planning/script/step2_generate_gpudrive_json.py \
        --metadata_path /data/llh/navsim_workspace/exp/pipeline_output/<TOKEN>/scene_metadata.pkl
"""
import sys
import os
import json
import pickle
import logging

import numpy as np

# Add ScenarioMax to import path
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

# Must match gpudrive/gpudrive/env/config.py  polyline_reduction_threshold
POLYLINE_REDUCTION_THRESHOLD = 0.1
# Must match gpudrive/src/consts.hpp  kMaxRoadEntityCount
MAX_ROAD_ENTITY_COUNT = 10000


def simplify_geometry(points, threshold):
    """Area-based iterative simplification matching GPUDrive json_serialization.hpp.

    Removes middle points of consecutive triplets whose triangle area is
    below *threshold*.  Repeats until no more points can be removed.
    """
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
            # find next two non-skipped points
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
    """Apply polyline simplification to all road features and report stats."""
    total_before = sum(len(r["geometry"]) for r in road_features)
    for road in road_features:
        road["geometry"] = simplify_geometry(road["geometry"], threshold)
    total_after = sum(len(r["geometry"]) for r in road_features)
    total_segments = sum(max(len(r["geometry"]) - 1, 0) for r in road_features)
    logger.info(
        f"  Simplification: {total_before} pts → {total_after} pts "
        f"({total_segments} segments)"
    )
    return road_features, total_segments


def convert_numpy(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
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


EPISODE_LEN = 91  # Must match madrona_gpudrive.episodeLen


def build_gpudrive_json(token, map_name, ego_heading, road_features):
    """Build a minimal GPUDrive scenario dict with a dummy ego and extracted roads.

    The dummy ego is placed at (0, 0) with the original heading, replicated
    across EPISODE_LEN timesteps to satisfy GPUDrive's C++ expectations.
    """
    pos_list = [{"x": 0.0, "y": 0.0, "z": 0.0}] * EPISODE_LEN
    heading_list = [ego_heading] * EPISODE_LEN
    vel_list = [{"x": 0.0, "y": 0.0}] * EPISODE_LEN
    valid_list = [True] * EPISODE_LEN

    return {
        "name": f"tfrecord-{token}.json",
        "scenario_id": token,
        "objects": [
            {
                "position": pos_list,
                "width": 1.8,
                "length": 4.049,
                "height": 1.5,
                "heading": heading_list,
                "velocity": vel_list,
                "valid": valid_list,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "type": "vehicle",
                "id": 0,
                "mark_as_expert": False,
            }
        ],
        "roads": road_features,
        "tl_states": {},
        "metadata": {
            "sdc_track_index": 0,
            "objects_of_interest": [],
            "tracks_to_predict": [
                {"track_index": 0, "difficulty": 0}
            ],
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 2: Generate GPUDrive JSON")
    parser.add_argument("--metadata_path", required=True, help="Path to scene_metadata.pkl from Step 1")
    args = parser.parse_args()

    # ── Load metadata ──
    with open(args.metadata_path, "rb") as f:
        metadata = pickle.load(f)

    token = metadata["token"]
    map_name = metadata["map_name"]
    ego_pose = metadata["ego_pose"]  # [x, y, heading]
    center = [float(ego_pose[0]), float(ego_pose[1])]
    ego_heading = float(ego_pose[2])

    logger.info(f"Token:      {token}")
    logger.info(f"Map:        {map_name}")
    logger.info(f"Center:     {center}")
    logger.info(f"Heading:    {ego_heading:.4f} rad")

    # ── Reconstruct map API ──
    logger.info(f"Loading map from {NUPLAN_MAPS_ROOT} ...")
    map_api = get_maps_api(NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, map_name)

    # ── ScenarioMax: extract static map elements ──
    logger.info("Extracting static map elements via ScenarioMax ...")
    static_map_elements = extract_static_map_elements(map_api, center)
    logger.info(f"  Extracted {len(static_map_elements)} map elements")

    # ── ScenarioMax: convert to GPUDrive road features ──
    logger.info("Converting to GPUDrive road format ...")
    road_features, edge_segments = convert_map_features(static_map_elements)

    if road_features is None:
        logger.error("3D structure detected — scenario rejected by convert_map_features")
        return

    logger.info(f"  Road features: {len(road_features)}")

    # ── Simplify geometry (matching GPUDrive C++ json_serialization.hpp) ──
    road_features, total_segments = simplify_road_features(
        road_features, POLYLINE_REDUCTION_THRESHOLD
    )
    if total_segments > MAX_ROAD_ENTITY_COUNT:
        logger.warning(
            f"  Total segments ({total_segments}) exceeds "
            f"kMaxRoadEntityCount ({MAX_ROAD_ENTITY_COUNT}). "
            f"Pruning distant roads."
        )
        # Sort roads by distance from ego (0,0), drop furthest
        for r in road_features:
            xs = [p["x"] for p in r["geometry"]]
            ys = [p["y"] for p in r["geometry"]]
            r["_dist"] = (sum(xs) / len(xs)) ** 2 + (sum(ys) / len(ys)) ** 2
        road_features.sort(key=lambda r: r["_dist"])
        # Greedily keep roads until segment budget exhausted
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

    # ── Build and write GPUDrive JSON ──
    scenario_dict = build_gpudrive_json(token, map_name, ego_heading, road_features)
    scenario_dict = convert_numpy(scenario_dict)

    output_dir = os.path.join(OUTPUT_BASE, token, "gpudrive_json")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tfrecord-{token}.json")

    with open(output_path, "w") as f:
        json.dump(scenario_dict, f)

    logger.info(f"Saved GPUDrive JSON: {output_path}")


if __name__ == "__main__":
    main()
