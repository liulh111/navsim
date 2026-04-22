"""
Step 3: Extract the full Waymo 2984-dim observation via GPUDrive (run in gpudrive env).

For each GPUDrive JSON from Step 2, loads a single-scene GPUDrive simulator
and extracts three native observation tensors **without normalisation** to
preserve physical units:

    ego_obs     (6,)      = [speed, length, width, rel_goal_x, rel_goal_y, is_collided]
    partner_obs (63, 6)   = per partner: [speed, rel_x, rel_y, orientation, length, width]
    road_obs    (200, 13) = [x, y, seg_len, seg_w, seg_h, orientation, type one-hot(7)]

    waymo_obs   (2984,)   = ego_obs ⊕ partner_obs.flatten() ⊕ road_obs.flatten()

Partner order is GPUDrive's native order (mirroring the standard
ScenarioMax → nuPlan → GPUDrive flow — no distance re-sort).

Usage:
    conda activate gpudrive
    python step3_extract_road_obs.py \\
        --json_dir <token1_json_dir> [--json_dir <token2_json_dir> ...]

    # or, auto-discover every tfrecord-*.json under OUTPUT_BASE:
    python step3_extract_road_obs.py --auto
"""
import sys
import os
import gc
import glob
import argparse
import logging
import subprocess
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from matplotlib import transforms
from matplotlib.patches import Rectangle

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.roadgraph import LocalRoadGraphPoints
from gpudrive.datatypes.observation import LocalEgoState, PartnerObs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_BASE = "/data/llh/navsim_workspace/exp/pipeline_output"

# ── Layout constants (must match navsim's Waymo whitebox convention) ──
EGO_DIM = 6
PARTNER_NUM = 63
PARTNER_DIM = 6
ROAD_NUM = 200
ROAD_DIM = 13
TOTAL_DIM = EGO_DIM + PARTNER_NUM * PARTNER_DIM + ROAD_NUM * ROAD_DIM  # 2984


ROAD_TYPE_NAMES = [
    "None", "RoadEdge", "RoadLine", "RoadLane",
    "CrossWalk", "SpeedBump", "StopSign",
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


# ── Observation extractors ──

def extract_ego_obs(env) -> np.ndarray:
    """(6,) = [speed, length, width, rel_goal_x, rel_goal_y, is_collided]."""
    ego_state = LocalEgoState.from_tensor(
        self_obs_tensor=env.sim.self_observation_tensor(),
        backend="torch",
        device="cpu",
    )
    stacked = torch.stack(
        [
            ego_state.speed,
            ego_state.vehicle_length,
            ego_state.vehicle_width,
            ego_state.rel_goal_x,
            ego_state.rel_goal_y,
            ego_state.is_collided,
        ],
        dim=-1,
    )  # (num_worlds, max_agents, 6)
    return stacked[0, 0].cpu().numpy().astype(np.float32)


def extract_partner_obs(env) -> np.ndarray:
    """(63, 6) per-partner [speed, rel_x, rel_y, orient, length, width]."""
    partner_obs = PartnerObs.from_tensor(
        partner_obs_tensor=env.sim.partner_observations_tensor(),
        backend="torch",
        device="cpu",
    )
    stacked = torch.cat(
        [
            partner_obs.speed,
            partner_obs.rel_pos_x,
            partner_obs.rel_pos_y,
            partner_obs.orientation,
            partner_obs.vehicle_length,
            partner_obs.vehicle_width,
        ],
        dim=-1,
    )  # (num_worlds, max_agents, num_partners, 6)
    arr = stacked[0, 0].cpu().numpy().astype(np.float32)  # (63, 6)
    if arr.shape[0] < PARTNER_NUM:
        pad = np.zeros((PARTNER_NUM - arr.shape[0], PARTNER_DIM), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:PARTNER_NUM]


def extract_road_obs(env) -> np.ndarray:
    """(200, 13) = [x, y, seg_len, seg_w, seg_h, orient, type one-hot(7)]."""
    roadgraph = LocalRoadGraphPoints.from_tensor(
        local_roadgraph_tensor=env.sim.agent_roadmap_tensor(),
        backend="torch",
        device="cpu",
    )
    roadgraph.one_hot_encode_road_point_types()
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
    return road_obs[0, 0].cpu().numpy().astype(np.float32)  # (200, 13)


def _draw_vehicle(ax, cx, cy, length, width, yaw, color, alpha=0.85, lw=1.2):
    """Draw an oriented rectangle + heading line at (cx, cy)."""
    rect = Rectangle(
        (-length / 2, -width / 2), length, width,
        linewidth=lw, edgecolor=color, facecolor="none", alpha=alpha,
    )
    t = transforms.Affine2D().rotate_around(0, 0, yaw).translate(cx, cy) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    head_x = cx + (length / 2) * np.cos(yaw)
    head_y = cy + (length / 2) * np.sin(yaw)
    ax.plot([cx, head_x], [cy, head_y], color=color, linewidth=lw + 0.5, alpha=alpha)


def visualize_road_obs(
    roads: np.ndarray,
    save_path: str,
    ego: np.ndarray = None,
    partners: np.ndarray = None,
    dpi: int = 160,
):
    """BEV visualisation of the 200x13 road block, optionally overlaying the
    ego box + goal and the 63 partner boxes. Everything is in ego-local frame
    (ego at origin, heading along +x)."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#1e1e1e")

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

    if ego is not None and ego.shape[-1] >= 6:
        _, ego_len, ego_wid, goal_x, goal_y, is_collided = ego[:6]
        ego_color = "red" if float(is_collided) > 0.5 else "lime"
        _draw_vehicle(ax, 0.0, 0.0, max(float(ego_len), 0.1),
                      max(float(ego_wid), 0.1), 0.0, color=ego_color, lw=2.0)
        ax.scatter([0], [0], c=ego_color, s=70, zorder=10, label="ego")
        ax.scatter([goal_x], [goal_y], c="yellow", marker="*", s=180, zorder=10, label="goal")
        ax.plot([0, goal_x], [0, goal_y], "--", color="yellow", alpha=0.6)
    else:
        ax.scatter([0], [0], c="lime", s=70, zorder=10, label="ego")

    n_partners_drawn = 0
    if partners is not None:
        for p in partners:
            if np.allclose(p, 0.0, atol=1e-6):
                continue
            n_partners_drawn += 1
            _, px, py, yaw, plen, pwid = p[:6]
            _draw_vehicle(ax, float(px), float(py),
                          max(float(plen), 0.1), max(float(pwid), 0.1),
                          float(yaw), color="#ea00ff", alpha=0.85, lw=1.2)
            ax.scatter([px], [py], c="#ea00ff", s=10, alpha=0.9)

    # Legend: road types + agent markers
    for name, color in ROAD_TYPE_COLORS.items():
        ax.scatter([], [], c=color, s=30, label=name)
    if partners is not None:
        ax.scatter([], [], c="#ea00ff", s=30, label=f"partner ({n_partners_drawn})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    title = f"GPUDrive Waymo Obs ({non_zero}/200 road"
    if partners is not None:
        title += f", {n_partners_drawn}/{PARTNER_NUM} partners"
    title += ")"
    ax.set_title(title)
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
    logger.info(f"Saved viz: {save_path}")


# ── Per-token pipeline ──

def process_one(json_dir: str, output_dir: str):
    logger.info("=" * 72)
    logger.info(f"JSON dir:   {json_dir}")
    logger.info(f"Output dir: {output_dir}")

    data_loader = SceneDataLoader(
        root=json_dir,
        batch_size=1,
        dataset_size=1,
        sample_with_replacement=False,
        file_prefix="tfrecord",
    )
    env_config = EnvConfig(
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=False,
        init_mode="all_valid",
    )

    logger.info("Initialising GPUDrive ...")
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=64,
        device="cpu",
        render_config=RenderConfig(),
    )

    try:
        ego_obs = extract_ego_obs(env)                    # (6,)
        partner_obs = extract_partner_obs(env)            # (63, 6)
        road_obs = extract_road_obs(env)                  # (200, 13)
    finally:
        env.close()

    waymo_obs = np.concatenate(
        [ego_obs, partner_obs.flatten(), road_obs.flatten()]
    ).astype(np.float32)
    assert waymo_obs.shape == (TOTAL_DIM,), waymo_obs.shape

    # Counts
    n_partners_valid = int(np.count_nonzero(np.any(partner_obs != 0, axis=1)))
    n_road_valid = int(np.count_nonzero(np.any(road_obs != 0, axis=1)))
    logger.info(
        f"ego speed={ego_obs[0]:.2f} m/s, goal=({ego_obs[3]:.2f}, {ego_obs[4]:.2f})"
    )
    logger.info(f"partners: {n_partners_valid}/{PARTNER_NUM} valid rows")
    logger.info(f"road:     {n_road_valid}/{ROAD_NUM} valid rows")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "ego_obs.npy"), ego_obs)
    np.save(os.path.join(output_dir, "partner_obs.npy"), partner_obs)
    np.save(os.path.join(output_dir, "road_obs_gpudrive.npy"), road_obs)
    np.save(os.path.join(output_dir, "waymo_obs_gpudrive.npy"), waymo_obs)
    logger.info("Saved: ego_obs.npy, partner_obs.npy, road_obs_gpudrive.npy, waymo_obs_gpudrive.npy")

    # # Full 2984-dim visualisation (ego + goal + partners + roads), all via
    # # the one enhanced visualize_road_obs helper.
    # full_png = os.path.join(output_dir, "waymo_obs_gpudrive.png")
    # visualize_road_obs(road_obs, full_png, ego=ego_obs, partners=partner_obs)

    # # Road-only debug visualisation (same helper, without ego/partners).
    # road_png = os.path.join(output_dir, "road_obs_gpudrive.png")
    # visualize_road_obs(road_obs, road_png)


def iter_auto_pairs(output_base: str):
    """Yield (json_dir, output_dir) discovered from OUTPUT_BASE in a streaming way."""
    pattern = os.path.join(output_base, "*", "gpudrive_json", "tfrecord-*.json")
    seen = set()
    for json_path in glob.iglob(pattern):
        json_dir = os.path.dirname(json_path)
        output_dir = os.path.dirname(json_dir)
        key = (os.path.abspath(json_dir), os.path.abspath(output_dir))
        if key in seen:
            continue
        seen.add(key)
        yield json_dir, output_dir


def run_one_in_subprocess(step3_script: str, json_dir: str, output_dir: str):
    """Run one scene in a fresh Python process so native memory is fully reclaimed."""
    cmd = [
        sys.executable,
        step3_script,
        "--json_dir", json_dir,
        "--output_dir", output_dir,
    ]
    logger.info("[isolated] Launching child process for one scene")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Child process failed (code={proc.returncode}) for {json_dir}")


def run_batch_in_subprocess(step3_script: str, batch_pairs: List[Tuple[str, str]]):
    """Run a small batch in one child process to reduce process startup overhead."""
    cmd = [sys.executable, step3_script]
    for json_dir, output_dir in batch_pairs:
        cmd.extend(["--json_dir", json_dir, "--output_dir", output_dir])
    logger.info(f"[isolated] Launching child process for batch size={len(batch_pairs)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Child batch failed (code={proc.returncode})")


def _run_batch_task(step3_script: str, batch_pairs: List[Tuple[str, str]], batch_idx: int):
    """Worker task for parallel isolated execution."""
    run_batch_in_subprocess(step3_script, batch_pairs)
    return batch_idx, len(batch_pairs)


def main():
    parser = argparse.ArgumentParser(description="Step 3: Extract full 2984-dim Waymo obs")
    parser.add_argument(
        "--json_dir", action="append", default=[],
        help="Directory with tfrecord-*.json from Step 2 (repeatable).",
    )
    parser.add_argument(
        "--output_dir", action="append", default=[],
        help="Output dir for .npy/.png (one per --json_dir, same order). "
             "If omitted, defaults to the parent of each json_dir.",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help=f"Auto-discover all gpudrive_json dirs under {OUTPUT_BASE}.",
    )
    parser.add_argument(
        "--isolated", action="store_true",
        help=(
            "Run each discovered scene in a fresh subprocess (recommended for --auto) "
            "to prevent native memory growth across thousands of scenes."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only process first N scenes after discovery/filtering (0 = no limit).",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip first N scenes after discovery/filtering.",
    )
    parser.add_argument(
        "--isolated_batch_size", type=int, default=16,
        help=(
            "When --isolated is enabled, process this many scenes per child process "
            "(1 = max isolation, higher = faster; default: 16)."
        ),
    )
    parser.add_argument(
        "--parallel_workers", type=int, default=1,
        help=(
            "Number of parent-side workers to execute isolated batches in parallel "
            "(effective only with --isolated --auto)."
        ),
    )
    args = parser.parse_args()

    if args.isolated_batch_size < 1:
        logger.error("--isolated_batch_size must be >= 1")
        sys.exit(1)
    if args.parallel_workers < 1:
        logger.error("--parallel_workers must be >= 1")
        sys.exit(1)

    explicit_pairs = []
    if args.auto:
        logger.info(f"Auto discovery root: {OUTPUT_BASE}")

    for i, json_dir in enumerate(args.json_dir):
        if i < len(args.output_dir):
            out = args.output_dir[i]
        else:
            out = os.path.dirname(json_dir)
        explicit_pairs.append((json_dir, out))

    # Build a streaming iterator to avoid accumulating all tasks in memory.
    def iter_all_pairs():
        seen = set()
        if args.auto:
            for jd, od in iter_auto_pairs(OUTPUT_BASE):
                key = (os.path.abspath(jd), os.path.abspath(od))
                if key in seen:
                    continue
                seen.add(key)
                yield jd, od
        for jd, od in explicit_pairs:
            key = (os.path.abspath(jd), os.path.abspath(od))
            if key in seen:
                continue
            seen.add(key)
            yield jd, od

    # Apply offset/limit lazily.
    count_seen = 0
    count_done = 0
    step3_script = os.path.abspath(__file__)

    has_any = False

    if args.isolated and args.auto:
        # Build batches first from a streaming source.
        batches: List[List[Tuple[str, str]]] = []
        pending_batch: List[Tuple[str, str]] = []
        for json_dir, output_dir in iter_all_pairs():
            has_any = True
            if count_seen < args.offset:
                count_seen += 1
                continue
            if args.limit > 0 and count_done >= args.limit:
                break

            pending_batch.append((json_dir, output_dir))
            count_seen += 1
            count_done += 1

            if len(pending_batch) >= args.isolated_batch_size:
                batches.append(pending_batch)
                pending_batch = []

        if pending_batch:
            batches.append(pending_batch)

        if args.parallel_workers == 1:
            for i, batch in enumerate(batches, start=1):
                logger.info(
                    f"\nProcessing batch {i}/{len(batches)} (size={len(batch)})"
                )
                run_batch_in_subprocess(step3_script, batch)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            logger.info(
                f"Running {len(batches)} isolated batches with parallel_workers={args.parallel_workers}"
            )
            futures = []
            with ThreadPoolExecutor(max_workers=args.parallel_workers) as ex:
                for i, batch in enumerate(batches, start=1):
                    futures.append(ex.submit(_run_batch_task, step3_script, batch, i))
                for fut in as_completed(futures):
                    batch_idx, batch_len = fut.result()
                    logger.info(
                        f"Completed batch {batch_idx}/{len(batches)} (size={batch_len})"
                    )
    else:
        for json_dir, output_dir in iter_all_pairs():
            has_any = True
            if count_seen < args.offset:
                count_seen += 1
                continue
            if args.limit > 0 and count_done >= args.limit:
                break

            logger.info(f"\nProcessing scene #{count_done + 1} (global #{count_seen + 1})")
            process_one(json_dir, output_dir)

            count_seen += 1
            count_done += 1

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not has_any:
        logger.error("No JSON dirs given. Use --json_dir or --auto.")
        sys.exit(1)

    logger.info(f"Done. Processed {count_done} scene(s).")


if __name__ == "__main__":
    main()
