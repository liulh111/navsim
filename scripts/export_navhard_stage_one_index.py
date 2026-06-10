#!/usr/bin/env python
"""Export NAVHARD Stage One current-frame metadata for ScenarioMax."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import yaml

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("OPENSCENE_DATA_ROOT", WORKSPACE_ROOT / "dataset"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "gpudrive_json" / "navhard_two_stage" / "stage_one_index.csv",
    )
    parser.add_argument(
        "--navsim-log-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "navsim_logs" / "test",
    )
    parser.add_argument(
        "--original-sensor-path",
        type=Path,
        default=DEFAULT_DATA_ROOT / "sensor_blobs" / "test",
    )
    parser.add_argument(
        "--nuplan-data-root",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "cache" / "test",
    )
    parser.add_argument("--tokens", nargs="*", help="Only export these Stage One tokens.")
    parser.add_argument("--limit", type=int, help="Only export the first N selected tokens.")
    return parser.parse_args()


def _load_scene_filter() -> SceneFilter:
    config_path = (
        WORKSPACE_ROOT
        / "navsim"
        / "navsim"
        / "planning"
        / "script"
        / "config"
        / "common"
        / "train_test_split"
        / "scene_filter"
        / "navhard_two_stage.yaml"
    )
    with config_path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)
    kwargs = {key: value for key, value in config.items() if not key.startswith("_")}
    scene_filter = SceneFilter(**kwargs)
    scene_filter.include_synthetic_scenes = False
    return scene_filter


def _resolve_db_path(nuplan_data_root: Path, log_name: str) -> Path:
    db_name = log_name if log_name.endswith(".db") else f"{log_name}.db"
    direct_path = nuplan_data_root / db_name
    if direct_path.is_file():
        return direct_path.resolve()

    matches = list(nuplan_data_root.rglob(db_name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one nuPlan DB named {db_name} under {nuplan_data_root}, found {len(matches)}"
        )
    return matches[0].resolve()


def main() -> None:
    args = _parse_args()
    scene_filter = _load_scene_filter()
    scene_loader = SceneLoader(
        data_path=args.navsim_log_dir,
        original_sensor_path=args.original_sensor_path,
        scene_filter=scene_filter,
        synthetic_sensor_path=None,
        synthetic_scenes_path=None,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    available_tokens = set(scene_loader.tokens_stage_one)
    selected_tokens = sorted(args.tokens if args.tokens else available_tokens)
    missing_tokens = sorted(set(selected_tokens) - available_tokens)
    if missing_tokens:
        raise ValueError(f"Not NAVHARD Stage One tokens: {missing_tokens[:10]}")
    if args.limit is not None:
        selected_tokens = selected_tokens[: args.limit]

    current_frame_index = scene_filter.num_history_frames - 1
    rows: list[dict[str, Any]] = []
    for token in selected_tokens:
        current_frame = scene_loader.scene_frames_dicts[token][current_frame_index]
        if current_frame["token"] != token:
            raise ValueError(
                f"Scene key {token} does not match current-frame token {current_frame['token']}"
            )
        log_name = str(current_frame["log_name"])
        rows.append(
            {
                "token": token,
                "timestamp": int(current_frame["timestamp"]),
                "log_name": log_name,
                "nuplan_db_path": str(_resolve_db_path(args.nuplan_data_root, log_name)),
                "map_name": str(current_frame["map_location"]),
                "current_frame_index": current_frame_index,
                "source_pkl": str((args.navsim_log_dir / f"{log_name}.pkl").resolve()),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token",
        "timestamp",
        "log_name",
        "nuplan_db_path",
        "map_name",
        "current_frame_index",
        "source_pkl",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} Stage One records to {args.output}")


if __name__ == "__main__":
    main()
