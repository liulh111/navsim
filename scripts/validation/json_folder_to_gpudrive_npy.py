#!/usr/bin/env python
"""Convert a folder of GPUDrive JSON files into one 2984-dim npy per scene."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp")


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


WORKSPACE_ROOT = _workspace_root()
GPUDRIVE_ROOT = WORKSPACE_ROOT / "gpudrive"
if str(GPUDRIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(GPUDRIVE_ROOT))

import numpy as np


TOKEN_RE = re.compile(r"([0-9a-fA-F]{16,})$")


@dataclass
class ListSceneDataLoader:
    scenes: Sequence[str]

    def __post_init__(self) -> None:
        self.batch_size = len(self.scenes)
        self.dataset_size = len(self.scenes)
        self._used = False

    def __iter__(self) -> Iterator[list[str]]:
        self._used = False
        return self

    def __next__(self) -> list[str]:
        if self._used:
            raise StopIteration
        self._used = True
        return list(self.scenes)

    def __len__(self) -> int:
        return 1


def _default_json_dir() -> Path:
    return WORKSPACE_ROOT / "results" / "gpudrive_json" / "navhard_two_stage"


def _default_output_dir(json_dir: Path) -> Path:
    if json_dir.name.endswith("_json"):
        return json_dir.with_name(json_dir.name[: -len("_json")] + "_npy")
    return json_dir.with_name(json_dir.name + "_npy")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=_default_json_dir())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--agent-index", type=int, default=0, help="Agent row to save as the 2984-dim observation.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--subprocess-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each batch in a fresh Python subprocess so GPUDrive/Madrona memory is released.",
    )
    parser.add_argument("--no-norm-obs", action="store_true")
    parser.add_argument("--obs-radius", type=float, default=50.0)
    parser.add_argument(
        "--road-obs-algorithm",
        default="linear",
        choices=("linear", "k_nearest_roadpoints"),
    )
    parser.add_argument(
        "--init-mode",
        default="all_non_trivial",
        choices=("all_non_trivial", "all_objects", "all_valid", "womd_tracks_to_predict"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-jsons", nargs="*", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-rows-path", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _token_from_json(path: Path) -> str:
    match = TOKEN_RE.search(path.stem)
    if match:
        return match.group(1).lower()
    return path.stem


def _json_files(json_dir: Path, max_scenes: int | None) -> list[Path]:
    files = sorted(path for path in json_dir.glob("*.json") if path.is_file())
    if max_scenes is not None:
        files = files[:max_scenes]
    if not files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")
    return files


def _make_env(scene_paths: Sequence[Path], args: argparse.Namespace) -> GPUDriveTorchEnv:
    from gpudrive.env.config import EnvConfig
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    config = EnvConfig(
        num_worlds=len(scene_paths),
        norm_obs=not args.no_norm_obs,
        obs_radius=args.obs_radius,
        road_obs_algorithm=args.road_obs_algorithm,
        init_mode=args.init_mode,
    )
    return GPUDriveTorchEnv(
        config=config,
        data_loader=ListSceneDataLoader([str(path) for path in scene_paths]),
        max_cont_agents=config.max_controlled_agents,
        device=args.device,
    )


def _export_batch(scene_paths: Sequence[Path], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    env = _make_env(scene_paths, args)
    try:
        obs = env.reset().detach().cpu().numpy().astype(np.float32, copy=False)
        controlled_mask = env.cont_agent_mask.detach().cpu().numpy().astype(np.bool_, copy=False)
    finally:
        env.close()
        del env
    return obs, controlled_mask


def _convert_files(files: Sequence[Path], args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        obs, controlled_mask = _export_batch(files, args)
        for offset, json_path in enumerate(files):
            token = _token_from_json(json_path)
            npy_path = output_dir / f"{token}.npy"
            np.save(npy_path, obs[offset, args.agent_index].astype(np.float32, copy=False))
            rows.append(
                {
                    "token": token,
                    "success": True,
                    "status": "exported",
                    "json_path": str(json_path),
                    "npy_path": str(npy_path),
                    "num_controlled_agents": int(controlled_mask[offset].sum()),
                }
            )
    except Exception as batch_error:  # noqa: BLE001
        print(f"[WARN] batch failed: {batch_error}", flush=True)
        for json_path in files:
            token = _token_from_json(json_path)
            npy_path = output_dir / f"{token}.npy"
            try:
                obs, controlled_mask = _export_batch([json_path], args)
                np.save(npy_path, obs[0, args.agent_index].astype(np.float32, copy=False))
                rows.append(
                    {
                        "token": token,
                        "success": True,
                        "status": "exported_single",
                        "json_path": str(json_path),
                        "npy_path": str(npy_path),
                        "num_controlled_agents": int(controlled_mask[0].sum()),
                    }
                )
            except Exception as scene_error:  # noqa: BLE001
                rows.append(
                    {
                        "token": token,
                        "success": False,
                        "status": repr(scene_error),
                        "json_path": str(json_path),
                        "npy_path": str(npy_path),
                        "num_controlled_agents": "",
                    }
                )
    return rows


def _run_worker(batch_files: Sequence[Path], args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    with tempfile.NamedTemporaryFile(prefix="gpudrive_npy_rows_", suffix=".json", delete=False) as temp_file:
        rows_path = Path(temp_file.name)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-rows-path",
        str(rows_path),
        "--json-dir",
        str(args.json_dir),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--agent-index",
        str(args.agent_index),
        "--obs-radius",
        str(args.obs_radius),
        "--road-obs-algorithm",
        str(args.road_obs_algorithm),
        "--init-mode",
        str(args.init_mode),
        "--worker-jsons",
        *[str(path) for path in batch_files],
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.no_norm_obs:
        cmd.append("--no-norm-obs")
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        rows_path.unlink(missing_ok=True)
        raise RuntimeError(f"Worker failed with exit code {result.returncode}")
    with rows_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    rows_path.unlink(missing_ok=True)
    return rows


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.agent_index < 0 or args.agent_index >= 64:
        raise ValueError("--agent-index must be in [0, 63]")

    output_dir = args.output_dir or _default_output_dir(args.json_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        if not args.worker_jsons or args.worker_rows_path is None:
            raise ValueError("Worker mode requires --worker-jsons and --worker-rows-path")
        rows = _convert_files([Path(path) for path in args.worker_jsons], args, output_dir)
        with args.worker_rows_path.open("w", encoding="utf-8") as file:
            json.dump(rows, file)
        return

    files = _json_files(args.json_dir, args.max_scenes)

    rows: list[dict[str, object]] = []
    for start in range(0, len(files), args.batch_size):
        end = min(start + args.batch_size, len(files))
        batch_files = []
        for index in range(start, end):
            token = _token_from_json(files[index])
            npy_path = output_dir / f"{token}.npy"
            if npy_path.is_file() and not args.overwrite:
                rows.append(
                    {
                        "token": token,
                        "success": True,
                        "status": "skipped_existing",
                        "json_path": str(files[index]),
                        "npy_path": str(npy_path),
                        "num_controlled_agents": "",
                    }
                )
                continue
            batch_files.append(files[index])

        if batch_files:
            if args.subprocess_batches:
                rows.extend(_run_worker(batch_files, args, output_dir))
            else:
                rows.extend(_convert_files(batch_files, args, output_dir))

        print(f"[{end}/{len(files)}] wrote {sum(bool(row['success']) for row in rows)} npy files", flush=True)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["token", "success", "status", "json_path", "npy_path", "num_controlled_agents"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {sum(bool(row['success']) for row in rows)}/{len(rows)} npy files to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
