#!/usr/bin/env python
"""Export ScenarioMax GPUDrive JSON for NAVHARD stage-1 tokens.

This script only does one thing: find NAVHARD stage-1 nuPlan-backed scenes and
call ScenarioMax's nuPlan exporter for those tokens.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import yaml
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


WORKSPACE_ROOT = _workspace_root()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "scenariomax_json" / "navhard_stage1",
    )
    parser.add_argument(
        "--navsim-log-dir",
        type=Path,
        default=Path(os.environ.get("OPENSCENE_DATA_ROOT", WORKSPACE_ROOT / "dataset")) / "navsim_logs" / "test",
    )
    parser.add_argument(
        "--original-sensor-path",
        type=Path,
        default=Path(os.environ.get("OPENSCENE_DATA_ROOT", WORKSPACE_ROOT / "dataset")) / "sensor_blobs" / "test",
    )
    parser.add_argument("--scenariomax-root", type=Path, default=WORKSPACE_ROOT / "ScenarioMax")
    parser.add_argument("--scenariomax-python", type=Path, default=None)
    parser.add_argument("--nuplan-data-root", type=Path, default=WORKSPACE_ROOT / "data" / "cache" / "test")
    parser.add_argument("--nuplan-maps-root", type=Path, default=WORKSPACE_ROOT / "dataset" / "maps")
    parser.add_argument("--dataset-version", default="navsim_token_eval")
    parser.add_argument("--tokens", nargs="*", default=None, help="Explicit token list.")
    parser.add_argument("--token-list", type=Path, default=None, help="Text file with one token per line.")
    parser.add_argument("--num-scenes", type=int, default=None, help="Randomly sample this many stage-1 scenes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _scene_filter_config() -> dict[str, Any]:
    cfg_path = (
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
    return _load_yaml(cfg_path)


def _discover_stage1_records(navsim_log_dir: Path, original_sensor_path: Path) -> dict[str, dict[str, Any]]:
    """Return token -> ScenarioMax export metadata for NAVSIM stage-1 scenes."""
    scene_filter_kwargs = {key: value for key, value in _scene_filter_config().items() if not key.startswith("_")}
    scene_filter = SceneFilter(**scene_filter_kwargs)
    scene_filter.include_synthetic_scenes = False
    scene_loader = SceneLoader(
        data_path=navsim_log_dir,
        original_sensor_path=original_sensor_path,
        scene_filter=scene_filter,
        synthetic_sensor_path=None,
        synthetic_scenes_path=None,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    current_index = scene_filter.num_history_frames - 1
    records: dict[str, dict[str, Any]] = {}
    for token in sorted(scene_loader.tokens_stage_one):
        frame_list = scene_loader.scene_frames_dicts[token]
        current = frame_list[current_index]
        records[token] = {
            "token": token,
            "log_name": current["log_name"],
            "timestamp": int(current["timestamp"]),
            "map_name": current["map_location"],
        }
    return dict(sorted(records.items()))


def _requested_tokens(args: argparse.Namespace, available_tokens: list[str]) -> list[str]:
    requested: list[str] | None = None
    if args.tokens:
        requested = list(args.tokens)
    if args.token_list:
        with args.token_list.open("r", encoding="utf-8") as file:
            from_file = [line.strip() for line in file if line.strip()]
        requested = (requested or []) + from_file
    if requested is None:
        requested = available_tokens

    missing = sorted(set(requested) - set(available_tokens))
    if missing:
        raise ValueError(f"Requested tokens are not NAVHARD stage-1 tokens: {missing[:10]}")

    selected = [token for token in available_tokens if token in set(requested)]
    if args.num_scenes is not None:
        rng = random.Random(args.seed)
        selected = rng.sample(selected, min(args.num_scenes, len(selected)))
        selected.sort()
    return selected


def _run_export(
    *,
    args: argparse.Namespace,
    scenariomax_python: Path,
    token: str,
    record: dict[str, Any],
) -> tuple[bool, str, Path]:
    output_path = args.output_dir / f"nuPlan_{args.dataset_version}_{token}.json"
    if output_path.is_file() and not args.overwrite:
        return True, "skipped_existing", output_path

    cmd = [
        str(scenariomax_python),
        str(args.scenariomax_root / "scripts" / "export_nuplan_token_obs.py"),
        "--nuplan-data-root",
        str(args.nuplan_data_root),
        "--nuplan-maps-root",
        str(args.nuplan_maps_root),
        "--log-name",
        str(record["log_name"]),
        "--token",
        token,
        "--timestamp",
        str(record["timestamp"]),
        "--map-name",
        str(record["map_name"]),
        "--dataset-version",
        args.dataset_version,
        "--output-dir",
        str(args.output_dir),
        "--export-mode",
        "json",
    ]
    result = subprocess.run(cmd, cwd=args.scenariomax_root, text=True, capture_output=True, check=False)
    log_path = args.output_dir / f"nuPlan_{args.dataset_version}_{token}.log"
    with log_path.open("w", encoding="utf-8") as file:
        file.write("$ " + " ".join(cmd) + "\n\nSTDOUT\n")
        file.write(result.stdout)
        file.write("\nSTDERR\n")
        file.write(result.stderr)
    if result.returncode != 0:
        return False, f"ScenarioMax failed; see {log_path}", output_path
    if not output_path.is_file():
        return False, f"ScenarioMax finished but missing output: {output_path}", output_path
    return True, "exported", output_path


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenariomax_python = args.scenariomax_python or (args.scenariomax_root / ".venv" / "bin" / "python")

    records = _discover_stage1_records(args.navsim_log_dir, args.original_sensor_path)
    selected_tokens = _requested_tokens(args, list(records))
    print(f"Discovered {len(records)} NAVHARD stage-1 scenes; selected {len(selected_tokens)}.", flush=True)
    rows: list[dict[str, Any]] = []
    for index, token in enumerate(selected_tokens, start=1):
        print(f"[{index}/{len(selected_tokens)}] {token}", flush=True)
        ok, status, output_path = _run_export(
            args=args,
            scenariomax_python=scenariomax_python,
            token=token,
            record=records[token],
        )
        rows.append(
            {
                **records[token],
                "success": ok,
                "status": status,
                "json_path": str(output_path),
            }
        )

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["token", "log_name", "timestamp", "map_name", "success", "status", "json_path"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {sum(row['success'] for row in rows)}/{len(rows)} JSON files to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
