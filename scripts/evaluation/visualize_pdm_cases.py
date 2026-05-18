#!/usr/bin/env python3
"""Select and visualize NAVSIM PDM evaluation cases."""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

NAVSIM_DEVKIT_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_DEVKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_DEVKIT_ROOT))
MATPLOTLIB_CACHE_DIR = Path("/tmp/navsim_matplotlib")
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))


METRIC_ALIASES = {
    "nc": "no_at_fault_collisions",
    "no_at_fault_collisions": "no_at_fault_collisions",
    "dac": "drivable_area_compliance",
    "drivable_area_compliance": "drivable_area_compliance",
    "ddc": "driving_direction_compliance",
    "driving_direction_compliance": "driving_direction_compliance",
    "tlc": "traffic_light_compliance",
    "traffic_light_compliance": "traffic_light_compliance",
    "ep": "ego_progress",
    "ego_progress": "ego_progress",
    "ttc": "time_to_collision_within_bound",
    "time_to_collision_within_bound": "time_to_collision_within_bound",
    "lk": "lane_keeping",
    "lane_keeping": "lane_keeping",
    "hc": "history_comfort",
    "history_comfort": "history_comfort",
    "ec": "two_frame_extended_comfort",
    "two_frame_extended_comfort": "two_frame_extended_comfort",
}

SUMMARY_SCORE_ALIASES = {"score", "summary", "pdm_score", "epdms", "extended_pdm_score"}


SORT_BY_HELP = """
sort_by values:
  Summary score:
    score, summary, pdm_score, epdms, extended_pdm_score

  Metric names and aliases:
    NC / no_at_fault_collisions
    DAC / drivable_area_compliance
    DDC / driving_direction_compliance
    TLC / traffic_light_compliance
    EP / ego_progress
    TTC / time_to_collision_within_bound
    LK / lane_keeping
    HC / history_comfort
    EC / two_frame_extended_comfort

  Exact CSV columns are also accepted, for example:
    ego_progress_stage_one
    ego_progress_stage_two
    time_to_collision_within_bound_stage_two
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select good/bad NAVSIM PDM cases from an evaluation CSV and render BEV "
            "visualizations with the agent trajectory. This script uses map/box/trajectory BEV only; "
            "it does not render camera images or LiDAR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SORT_BY_HELP,
    )
    parser.add_argument("--stage", required=True, choices=["1", "2", "stage1", "stage2"], help="Stage to inspect.")
    parser.add_argument(
        "--quality",
        required=True,
        choices=["good", "bad", "best", "worst"],
        help="Select high-scoring cases or low-scoring cases.",
    )
    parser.add_argument("--num_scenes", "-n", required=True, type=int, help="Number of scenes to visualize.")
    parser.add_argument("--csv_path", required=True, type=Path, help="Path to the PDM evaluation CSV.")
    parser.add_argument(
        "--sort_by",
        default="score",
        help=(
            "Metric used for ranking. See the list of accepted values below."
        ),
    )
    parser.add_argument("--output_dir", required=True, type=Path, help="Folder for visualizations and selected_scenes.csv.")
    parser.add_argument(
        "--viz_type",
        default="both",
        choices=["static", "dynamic", "both"],
        help=(
            "static writes PNG, dynamic writes GIF, both writes both. Dynamic GIFs reuse NAVSIM's "
            "frame_plot_to_gif flow and render per-frame scene map/boxes without camera images or LiDAR."
        ),
    )

    parser.add_argument(
        "--train_test_split",
        default="navhard_two_stage",
        help="Hydra train_test_split config used to load scenes.",
    )
    parser.add_argument("--agent", default="constant_velocity_agent", help="Hydra agent config used to recompute paths.")
    parser.add_argument(
        "--draw_gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw ground-truth future trajectory for stage 1 scenes.",
    )
    parser.add_argument(
        "--config_name",
        default="default_run_pdm_score",
        help="Hydra PDM scoring config to compose.",
    )
    return parser.parse_args()


def normalize_stage(stage: str) -> Tuple[int, str]:
    stage_id = 1 if stage in {"1", "stage1"} else 2
    return stage_id, "stage_one" if stage_id == 1 else "stage_two"


def normalize_quality(quality: str) -> bool:
    """Return True when larger values are better."""
    return quality in {"good", "best"}


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = [name for name in (reader.fieldnames or []) if name]
        rows = [{k: v for k, v in row.items() if k} for row in reader]
    return fieldnames, rows


def resolve_sort_column(sort_by: str, stage_suffix: str, fieldnames: Sequence[str]) -> str:
    key = sort_by.strip()
    key_lower = key.lower()

    if key in fieldnames:
        return key
    if key_lower in SUMMARY_SCORE_ALIASES:
        return "score"

    metric_base = METRIC_ALIASES.get(key_lower, key)
    staged_column = f"{metric_base}_{stage_suffix}"
    if staged_column in fieldnames:
        return staged_column

    supported = ["score"] + sorted(METRIC_ALIASES)
    raise ValueError(
        f"Cannot resolve sort_by='{sort_by}'. Use an exact CSV column or one of: {', '.join(supported)}"
    )


def select_rows(
    rows: Iterable[Dict[str, str]],
    stage_suffix: str,
    sort_column: str,
    larger_is_better: bool,
    num_scenes: int,
) -> List[Dict[str, str]]:
    stage_marker = f"no_at_fault_collisions_{stage_suffix}"
    candidates = []

    for row in rows:
        token = row.get("token", "")
        if token.startswith("extended_pdm_score"):
            continue
        if parse_float(row.get(stage_marker)) is None:
            continue
        sort_value = parse_float(row.get(sort_column))
        if sort_value is None:
            continue
        candidates.append((sort_value, row))

    candidates.sort(key=lambda item: item[0], reverse=larger_is_better)
    selected = []
    for rank, (sort_value, row) in enumerate(candidates[:num_scenes], start=1):
        row = dict(row)
        row["rank"] = str(rank)
        row["sort_column"] = sort_column
        row["sort_value"] = f"{sort_value:.8f}"
        selected.append(row)
    return selected


def compose_cfg(config_name: str, train_test_split: str, agent_name: str, tokens: List[str], stage_id: int) -> Any:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    hydra.initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None)
    cfg = hydra.compose(
        config_name=config_name,
        overrides=[f"train_test_split={train_test_split}", f"agent={agent_name}"],
    )

    with open_dict(cfg):
        scene_filter = cfg.train_test_split.scene_filter
        scene_filter.max_scenes = None
        if stage_id == 1:
            scene_filter.tokens = tokens
            scene_filter.include_synthetic_scenes = False
            scene_filter.synthetic_scene_tokens = None
        else:
            scene_filter.tokens = []
            scene_filter.include_synthetic_scenes = True
            scene_filter.synthetic_scene_tokens = tokens
    return cfg


def build_scene_loader(cfg: Any, sensor_config: Any) -> Any:
    from hydra.utils import instantiate

    from navsim.common.dataloader import SceneLoader

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    return SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def set_last_line_label(ax: Any, label: str) -> None:
    if ax.lines:
        ax.lines[-1].set_label(label)


def compute_agent_trajectory(scene: Any, scene_loader: Any, agent: Any, token: str) -> Any:
    agent_input = scene_loader.get_agent_input_from_token(token)
    if getattr(agent, "requires_scene", False):
        return agent.compute_trajectory(agent_input, scene)
    return agent.compute_trajectory(agent_input)


def get_ground_truth_trajectory(scene: Any, stage_id: int, draw_gt: bool) -> Optional[Any]:
    if stage_id != 1 or not draw_gt:
        return None
    try:
        return scene.get_future_trajectory(num_trajectory_frames=scene.scene_metadata.num_future_frames)
    except Exception as exc:
        print(f"[warn] Could not get ground truth for token={scene.scene_metadata.initial_token}: {exc}")
        return None


def visualize_case(
    row: Dict[str, str],
    stage_id: int,
    scene_loader: Any,
    agent: Any,
    output_dir: Path,
    draw_gt: bool,
) -> Optional[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
    from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
    from navsim.visualization.plots import configure_ax, configure_bev_ax

    token = row["token"]
    scene = scene_loader.get_scene_from_token(token)
    agent_trajectory = compute_agent_trajectory(scene, scene_loader, agent, token)
    gt_trajectory = get_ground_truth_trajectory(scene, stage_id, draw_gt)

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

    if gt_trajectory is not None:
        add_trajectory_to_bev_ax(ax, gt_trajectory, TRAJECTORY_CONFIG["human"])
        set_last_line_label(ax, "ground truth")

    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    set_last_line_label(ax, "agent")

    configure_bev_ax(ax)
    configure_ax(ax)
    ax.set_title(
        f"stage {stage_id} rank {row['rank']} | {row['sort_column']}={row['sort_value']} | score={row.get('score', '')}",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    filename = (
        f"{int(row['rank']):03d}_stage{stage_id}_{safe_name(row['sort_column'])}_"
        f"{safe_name(row['sort_value'])}_{safe_name(token)}.png"
    )
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def plot_trajectory_prefix(ax: Any, trajectory: Any, config: Dict[str, Any], num_poses: int, label: str) -> None:
    poses = [[0.0, 0.0]] + trajectory.poses[:num_poses, :2].tolist()
    ax.plot(
        [pose[1] for pose in poses],
        [pose[0] for pose in poses],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
        label=label,
    )


def plot_global_state_prefix(
    ax: Any,
    origin: Any,
    initial_state: Any,
    future_states: Sequence[Any],
    config: Dict[str, Any],
    num_poses: int,
    label: str,
) -> None:
    import numpy as np

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    states = [initial_state] + list(future_states[:num_poses])
    state_array = np.array([[state.x, state.y, state.heading] for state in states], dtype=np.float64)
    local_states = convert_absolute_to_relative_se2_array(origin, state_array)
    ax.plot(
        local_states[:, 1],
        local_states[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
        label=label,
    )


def visualize_case_dynamic(
    row: Dict[str, str],
    stage_id: int,
    scene_loader: Any,
    agent: Any,
    output_dir: Path,
    draw_gt: bool,
) -> Optional[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from nuplan.common.actor_state.car_footprint import CarFootprint
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.geometry.convert import relative_to_absolute_poses

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )
    from navsim.visualization.bev import add_annotations_to_bev_ax, add_map_to_bev_ax, add_oriented_box_to_bev_ax
    from navsim.visualization.config import AGENT_CONFIG, BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
    from navsim.visualization.plots import configure_ax, configure_bev_ax
    from navsim.visualization.plots import frame_plot_to_gif

    token = row["token"]
    scene = scene_loader.get_scene_from_token(token)
    agent_trajectory = compute_agent_trajectory(scene, scene_loader, agent, token)
    gt_trajectory = get_ground_truth_trajectory(scene, stage_id, draw_gt)

    max_poses = len(agent_trajectory.poses)
    if gt_trajectory is not None:
        max_poses = max(max_poses, len(gt_trajectory.poses))

    current_frame_idx = scene.scene_metadata.num_history_frames - 1
    last_frame_idx = min(len(scene.frames) - 1, current_frame_idx + max_poses)
    frame_indices = list(range(current_frame_idx, last_frame_idx + 1))

    current_ego_pose = StateSE2(*scene.frames[current_frame_idx].ego_status.ego_pose)
    relative_states = [StateSE2(*pose) for pose in agent_trajectory.poses]
    predicted_global_states = relative_to_absolute_poses(current_ego_pose, relative_states)
    gt_global_states = [
        StateSE2(*scene.frames[idx].ego_status.ego_pose)
        for idx in range(current_frame_idx + 1, min(len(scene.frames), current_frame_idx + 1 + max_poses))
    ]

    def predicted_ego_pose_in_frame(frame_idx: int) -> StateSE2:
        prediction_idx = frame_idx - current_frame_idx - 1
        if prediction_idx < 0:
            return StateSE2(0.0, 0.0, 0.0)
        prediction_idx = min(prediction_idx, len(predicted_global_states) - 1)
        predicted_global_state = predicted_global_states[prediction_idx]
        frame_ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
        local_state = convert_absolute_to_relative_se2_array(
            frame_ego_pose,
            np.array([[predicted_global_state.x, predicted_global_state.y, predicted_global_state.heading]], dtype=np.float64),
        )[0]
        return StateSE2(*local_state)

    def plot_scene_frame_with_agent(scene_to_plot: Any, frame_idx: int) -> Tuple[Any, Any]:
        frame = scene_to_plot.frames[frame_idx]
        fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
        add_map_to_bev_ax(ax, scene_to_plot.map_api, StateSE2(*frame.ego_status.ego_pose))
        add_annotations_to_bev_ax(ax, frame.annotations, add_ego=False)

        predicted_ego_box = CarFootprint.build_from_rear_axle(
            rear_axle_pose=predicted_ego_pose_in_frame(frame_idx),
            vehicle_parameters=get_pacifica_parameters(),
        ).oriented_box
        add_oriented_box_to_bev_ax(
            ax,
            predicted_ego_box,
            AGENT_CONFIG[TrackedObjectType.EGO],
            add_heading=False,
        )

        pose_count = max(0, frame_idx - current_frame_idx)
        frame_ego_pose = StateSE2(*frame.ego_status.ego_pose)

        if gt_trajectory is not None:
            plot_global_state_prefix(
                ax,
                frame_ego_pose,
                current_ego_pose,
                gt_global_states,
                TRAJECTORY_CONFIG["human"],
                min(pose_count, len(gt_trajectory.poses)),
                "ground truth",
            )
        plot_global_state_prefix(
            ax,
            frame_ego_pose,
            current_ego_pose,
            predicted_global_states,
            TRAJECTORY_CONFIG["agent"],
            min(pose_count, len(agent_trajectory.poses)),
            "agent",
            )

        configure_bev_ax(ax)
        configure_ax(ax)
        ax.set_title(
            (
                f"stage {stage_id} rank {row['rank']} | t={pose_count * 0.5:.1f}s | "
                f"{row['sort_column']}={row['sort_value']}"
            ),
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        return fig, ax

    filename = (
        f"{int(row['rank']):03d}_stage{stage_id}_{safe_name(row['sort_column'])}_"
        f"{safe_name(row['sort_value'])}_{safe_name(token)}.gif"
    )
    output_path = output_dir / filename
    frame_plot_to_gif(str(output_path), plot_scene_frame_with_agent, scene, frame_indices, duration=350)
    return str(output_path)


def write_selected_csv(output_path: Path, rows: List[Dict[str, str]], extra_fields: Sequence[str]) -> None:
    if not rows:
        return
    preferred_fields = ["rank", "token", "valid", "score", "sort_column", "sort_value"]
    fields = preferred_fields + [field for field in extra_fields if field not in preferred_fields]
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    stage_id, stage_suffix = normalize_stage(args.stage)
    larger_is_better = normalize_quality(args.quality)

    fieldnames, rows = read_csv_rows(args.csv_path)
    sort_column = resolve_sort_column(args.sort_by, stage_suffix, fieldnames)
    selected_rows = select_rows(rows, stage_suffix, sort_column, larger_is_better, args.num_scenes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_selected_csv(args.output_dir / "selected_scenes.csv", selected_rows, fieldnames)

    if not selected_rows:
        print("No matching scenes were found.")
        return

    tokens = [row["token"] for row in selected_rows]
    cfg = compose_cfg(args.config_name, args.train_test_split, args.agent, tokens, stage_id)
    from hydra.utils import instantiate

    agent = instantiate(cfg.agent)
    agent.initialize()
    scene_loader = build_scene_loader(cfg, agent.get_sensor_config())

    rendered_paths = []
    animation_paths = []
    skipped = []
    for row in selected_rows:
        try:
            if args.viz_type in {"static", "both"}:
                path = visualize_case(row, stage_id, scene_loader, agent, args.output_dir, args.draw_gt)
                if path is not None:
                    rendered_paths.append(path)
                    row["visualization_path"] = path
            if args.viz_type in {"dynamic", "both"}:
                animation_path = visualize_case_dynamic(row, stage_id, scene_loader, agent, args.output_dir, args.draw_gt)
                if animation_path is not None:
                    animation_paths.append(animation_path)
                    row["animation_path"] = animation_path
        except Exception as exc:
            message = f"{row['token']}: {exc}"
            skipped.append(message)
            print(f"[warn] Failed to visualize {message}")

    write_selected_csv(
        args.output_dir / "selected_scenes.csv",
        selected_rows,
        list(fieldnames) + ["visualization_path", "animation_path"],
    )

    if skipped:
        (args.output_dir / "skipped.txt").write_text("\n".join(skipped) + "\n")

    print(f"Selected {len(selected_rows)} scenes using {sort_column}.")
    print(f"Rendered {len(rendered_paths)} static PNG visualizations to {args.output_dir}.")
    print(f"Rendered {len(animation_paths)} dynamic GIF visualizations to {args.output_dir}.")
    if skipped:
        print(f"Skipped {len(skipped)} scenes; see {args.output_dir / 'skipped.txt'}.")


if __name__ == "__main__":
    main()
