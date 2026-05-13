#!/usr/bin/env python
"""Visualize GPUDrive 2984-dim npy observations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OBS_DIM = 2984
MAX_AGENTS = 64
PARTNER_COUNT = 63
ROAD_COUNT = 200
EGO_DIM = 6
PARTNER_DIM = 6
ROAD_DIM = 13

MAX_SPEED = 100.0
MAX_VEH_LEN = 30.0
MAX_VEH_WIDTH = 15.0
MAX_REL_COORD = 1000.0
MAX_RG_COORD = 1000.0
MAX_ROAD_LINE_SEGMENT_LEN = 100.0
MAX_ROAD_SCALE = 100.0
MAX_ORIENTATION_RAD = 2.0 * np.pi

ROAD_TYPE_NAMES = ("none", "RoadLine", "RoadEdge", "RoadLane", "CrossWalk", "SpeedBump", "StopSign")
ROAD_COLORS = {
    0: "#b8b8b8",
    1: "#f2c94c",
    2: "#4f4f4f",
    3: "#2f80ed",
    4: "#27ae60",
    5: "#eb5757",
    6: "#9b51e0",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="A .npy file or a folder containing .npy files.")
    parser.add_argument("--agent-index", type=int, default=0, help="Used when a npy has shape (64, 2984).")
    parser.add_argument("--raw-obs", action="store_true", help="Input obs is not GPUDrive-normalized.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _iter_npy_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for npy_path in sorted(path.rglob("*.npy")):
        if npy_path.name in {"controlled_agent_mask.npy", "tokens.npy", "json_paths.npy", "num_controlled_agents.npy", "success_mask.npy"}:
            continue
        files.append(npy_path)
    if not files:
        raise FileNotFoundError(f"No observation npy files found under {path}")
    return files


def _select_obs(path: Path, agent_index: int) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.shape == (OBS_DIM,):
        return np.asarray(arr, dtype=np.float64)
    if arr.shape == (MAX_AGENTS, OBS_DIM):
        return np.asarray(arr[agent_index], dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == OBS_DIM:
        return np.asarray(arr[0], dtype=np.float64)
    raise ValueError(f"Unsupported npy shape for visualization: {path} {arr.shape}")


def _parse_obs(obs: np.ndarray, normalized: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ego = obs[:EGO_DIM].copy()
    partner_start = EGO_DIM
    road_start = partner_start + PARTNER_COUNT * PARTNER_DIM
    partners = obs[partner_start:road_start].reshape(PARTNER_COUNT, PARTNER_DIM).copy()
    roads = obs[road_start:].reshape(ROAD_COUNT, ROAD_DIM).copy()
    road_numeric = roads[:, :6].copy()
    road_type = np.argmax(roads[:, 6:], axis=1)

    if normalized:
        ego[0] *= MAX_SPEED
        ego[1] *= MAX_VEH_LEN
        ego[2] *= MAX_VEH_WIDTH
        ego[3] *= MAX_REL_COORD
        ego[4] *= MAX_REL_COORD
        partners[:, 0] *= MAX_SPEED
        partners[:, 1] *= MAX_REL_COORD
        partners[:, 2] *= MAX_REL_COORD
        partners[:, 3] *= MAX_ORIENTATION_RAD
        partners[:, 4] *= MAX_VEH_LEN
        partners[:, 5] *= MAX_VEH_WIDTH
        road_numeric[:, 0] *= MAX_RG_COORD
        road_numeric[:, 1] *= MAX_RG_COORD
        road_numeric[:, 2] *= MAX_ROAD_LINE_SEGMENT_LEN
        road_numeric[:, 3] *= MAX_ROAD_SCALE
        road_numeric[:, 4] *= MAX_ROAD_SCALE
        road_numeric[:, 5] *= MAX_ORIENTATION_RAD
    return ego, partners, road_numeric, road_type


def _valid_partners(partners: np.ndarray) -> np.ndarray:
    return (np.abs(partners[:, 4]) > 1e-6) | (np.abs(partners[:, 5]) > 1e-6)


def _valid_roads(roads: np.ndarray, road_type: np.ndarray) -> np.ndarray:
    return np.any(np.abs(roads) > 1e-6, axis=1) | (road_type != 0)


def _box_corners(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    forward = np.array([np.cos(heading), np.sin(heading)]) * (length / 2.0)
    side = np.array([-np.sin(heading), np.cos(heading)]) * (width / 2.0)
    center = np.array([x, y])
    return np.array([center + forward + side, center + forward - side, center - forward - side, center - forward + side])


def _draw_box(ax, x: float, y: float, heading: float, length: float, width: float, color: str, label: str | None = None) -> None:
    corners = _box_corners(x, y, heading, max(length, 0.1), max(width, 0.1))
    ax.add_patch(plt.Polygon(corners, closed=True, facecolor=color, edgecolor="black", alpha=0.65, linewidth=0.8, label=label))
    ax.plot([x, x + np.cos(heading) * max(length / 2.0, 0.5)], [y, y + np.sin(heading) * max(length / 2.0, 0.5)], color="black", linewidth=0.8)


def _draw_road(ax, x: float, y: float, heading: float, half_len: float, road_type: int) -> None:
    direction = np.array([np.cos(heading), np.sin(heading)])
    start = np.array([x, y]) - direction * half_len
    end = np.array([x, y]) + direction * half_len
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=ROAD_COLORS.get(int(road_type), "#b8b8b8"),
        linewidth=2.0 if road_type in (2, 3) else 1.2,
        alpha=0.75,
    )


def _visualize_one(npy_path: Path, args: argparse.Namespace) -> Path:
    obs = _select_obs(npy_path, args.agent_index)
    ego, partners, roads, road_type = _parse_obs(obs, normalized=not args.raw_obs)
    output_path = npy_path.with_suffix(".png")
    if output_path.exists() and not args.overwrite:
        return output_path

    fig, ax = plt.subplots(figsize=(10, 10))
    road_idx = np.where(_valid_roads(roads, road_type))[0]
    for idx in road_idx:
        _draw_road(ax, roads[idx, 0], roads[idx, 1], roads[idx, 5], roads[idx, 2], int(road_type[idx]))

    partner_idx = np.where(_valid_partners(partners))[0]
    for idx in partner_idx:
        _, x, y, heading, length, width = partners[idx]
        _draw_box(ax, x, y, heading, length, width, "#f2994a")

    ego_length = ego[1] if ego[1] > 0 else 4.8
    ego_width = ego[2] if ego[2] > 0 else 2.0
    _draw_box(ax, 0.0, 0.0, 0.0, ego_length, ego_width, "#56ccf2", label="ego")
    ax.scatter([ego[3]], [ego[4]], marker="*", s=160, c="#eb5757", edgecolors="black", linewidths=0.7, label="goal")

    for type_id in sorted(set(int(road_type[idx]) for idx in road_idx)):
        ax.plot([], [], color=ROAD_COLORS.get(type_id, "#b8b8b8"), label=f"road:{ROAD_TYPE_NAMES[type_id]}")

    points = [[0.0, 0.0], [ego[3], ego[4]]]
    if len(partner_idx):
        points.extend(partners[partner_idx, 1:3].tolist())
    if len(road_idx):
        points.extend(roads[road_idx, :2].tolist())
    points_arr = np.asarray(points, dtype=np.float64)
    min_xy = np.nanmin(points_arr, axis=0)
    max_xy = np.nanmax(points_arr, axis=0)
    center = (min_xy + max_xy) / 2.0
    span = max(float(np.max(max_xy - min_xy)), 20.0)
    pad = span * 0.12
    ax.set_xlim(center[0] - span / 2.0 - pad, center[0] + span / 2.0 + pad)
    ax.set_ylim(center[1] - span / 2.0 - pad, center[1] + span / 2.0 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_xlabel("ego-frame x (m)")
    ax.set_ylabel("ego-frame y (m)")
    ax.set_title(npy_path.stem)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    args = _parse_args()
    files = _iter_npy_files(args.path)
    for index, npy_path in enumerate(files, start=1):
        try:
            output_path = _visualize_one(npy_path, args)
            print(f"[{index}/{len(files)}] {output_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {npy_path}: {exc}")


if __name__ == "__main__":
    main()
