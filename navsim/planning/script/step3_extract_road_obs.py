"""
Step 3: Extract road observations via GPUDrive (run in gpudrive env).

Loads the GPUDrive JSON from Step 2, initialises a GPUDrive simulator,
extracts the 200x13 road observation tensor, saves it as .npy, and
produces a BEV visualisation.

Usage:
    conda activate gpudrive
    python /data/llh/navsim_workspace/navsim/navsim/planning/script/step3_extract_road_obs.py \
        --json_dir /data/llh/navsim_workspace/exp/pipeline_output/<TOKEN>/gpudrive_json \
        --output_dir /data/llh/navsim_workspace/exp/pipeline_output/<TOKEN>
"""
import sys
import os
import unittest.mock
import argparse
import logging

# Patch jaxlib import issue in gpudrive env.
# ArrayImpl must be a real type so isinstance() checks don't crash.
_fake_mod = type(sys)("jaxlib.xla_extension")

class _FakeArrayImpl:
    pass

_fake_mod.ArrayImpl = _FakeArrayImpl
sys.modules["jaxlib.xla_extension"] = _fake_mod

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.roadgraph import LocalRoadGraphPoints

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Road type names matching GPUDrive EntityType enum (0-6) ──
ROAD_TYPE_NAMES = [
    "None",       # 0
    "RoadEdge",   # 1
    "RoadLine",   # 2
    "RoadLane",   # 3
    "CrossWalk",  # 4
    "SpeedBump",  # 5
    "StopSign",   # 6
]
ROAD_TYPE_COLORS = {
    "None":      "#999999",
    "RoadEdge":  "#ff8800",
    "RoadLine":  "#ffffff",
    "RoadLane":  "#00bfff",
    "CrossWalk": "#00ff7f",
    "SpeedBump": "#ff00ff",
    "StopSign":  "#ff3333",
}


def extract_road_obs(env) -> np.ndarray:
    """Extract 200x13 road observations from GPUDrive, matching the pipeline exactly.

    Uses the same code path as env_torch._get_road_map_obs(mask=None),
    but returns the un-normalised, un-flattened (200, 13) array for the
    ego agent in world 0.
    """
    roadgraph = LocalRoadGraphPoints.from_tensor(
        local_roadgraph_tensor=env.sim.agent_roadmap_tensor(),
        backend="torch",
        device="cpu",
    )
    # One-hot encode: type int (0-6) → 7-dim one-hot, matching GPUDrive exactly
    roadgraph.one_hot_encode_road_point_types()

    # Assemble (num_worlds, num_agents, 200, 13)
    road_obs = torch.cat(
        [
            roadgraph.x.unsqueeze(-1),
            roadgraph.y.unsqueeze(-1),
            roadgraph.segment_length.unsqueeze(-1),
            roadgraph.segment_width.unsqueeze(-1),
            roadgraph.segment_height.unsqueeze(-1),
            roadgraph.orientation.unsqueeze(-1),
            roadgraph.type.float(),
        ],
        dim=-1,
    )

    # Take ego agent (first agent in first world)
    ego_road = road_obs[0, 0].numpy()  # (200, 13)
    return ego_road


def visualize_road_obs(roads: np.ndarray, save_path: str, dpi: int = 160):
    """BEV visualisation of 200x13 road observations."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#1e1e1e")

    # Draw ego marker
    ax.scatter([0], [0], c="lime", s=80, zorder=10, label="ego")

    non_zero = 0
    for r in roads:
        if np.allclose(r, 0.0, atol=1e-6):
            continue
        non_zero += 1
        x, y, seg_len, _, _, ori = r[:6]
        onehot = r[6:13]
        t_idx = int(np.argmax(onehot)) if np.sum(onehot) > 0 else 0
        name = ROAD_TYPE_NAMES[t_idx] if t_idx < len(ROAD_TYPE_NAMES) else "None"
        color = ROAD_TYPE_COLORS.get(name, "#999999")

        ax.scatter([x], [y], c=color, s=6, alpha=0.9)
        dx = float(seg_len) * np.cos(float(ori))
        dy = float(seg_len) * np.sin(float(ori))
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, alpha=0.4, linewidth=1.0)

    # Legend
    for name, color in ROAD_TYPE_COLORS.items():
        ax.scatter([], [], c=color, s=30, label=name)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    ax.set_title(f"GPUDrive Road Observations ({non_zero}/200 non-zero)")
    ax.set_xlabel("x (ego-local)")
    ax.set_ylabel("y (ego-local)")
    ax.axis("equal")
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved visualisation: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 3: Extract road obs via GPUDrive")
    parser.add_argument("--json_dir", required=True, help="Directory with tfrecord-*.json from Step 2")
    parser.add_argument("--output_dir", required=True, help="Directory for output .npy and .png")
    args = parser.parse_args()

    # ── Data loader ──
    data_loader = SceneDataLoader(
        root=args.json_dir,
        batch_size=1,
        dataset_size=1,
        sample_with_replacement=False,
        file_prefix="tfrecord",
    )

    # ── Env config (unnormalised for direct comparison) ──
    env_config = EnvConfig(
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=False,
    )

    # ── Initialise GPUDrive ──
    logger.info("Initialising GPUDrive ...")
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=1,
        device="cpu",
        render_config=RenderConfig(),
    )

    # ── Extract road observations ──
    logger.info("Extracting road observations ...")
    road_obs = extract_road_obs(env)  # (200, 13)
    non_zero = np.count_nonzero(np.any(road_obs != 0, axis=1))
    logger.info(f"Road obs shape: {road_obs.shape}, non-zero rows: {non_zero}/200")

    # ── Save ──
    os.makedirs(args.output_dir, exist_ok=True)

    npy_path = os.path.join(args.output_dir, "road_obs_gpudrive.npy")
    np.save(npy_path, road_obs)
    logger.info(f"Saved road obs: {npy_path}")

    png_path = os.path.join(args.output_dir, "road_obs_gpudrive.png")
    visualize_road_obs(road_obs, png_path)

    env.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
