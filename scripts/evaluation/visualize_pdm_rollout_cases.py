#!/usr/bin/env python3
"""Render evaluation-aligned NAVSIM PDM rollout GIFs for selected cases."""

import argparse
import csv
import inspect
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
            "Select NAVSIM PDM cases from an evaluation CSV and render GIFs aligned with the "
            "evaluation rollout: agent trajectory -> PDMSimulator -> reactive traffic policy."
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
    parser.add_argument("--sort_by", default="score", help="Metric used for ranking. See accepted values below.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Folder for GIFs and selected_scenes.csv.")
    parser.add_argument("--train_test_split", default="navhard_two_stage", help="Hydra train_test_split config.")
    parser.add_argument("--agent", default="token_obs_mlp_agent", help="Hydra agent config to recompute paths.")
    parser.add_argument(
        "--config_name",
        default="default_run_token_obs_mlp_pdm_score",
        help="Hydra PDM scoring config.",
    )
    parser.add_argument(
        "--agent_obs_dir",
        default="/data/llh/navsim_workspace/results/gpudrive_json/navhard_two_stage_npy",
        help="TokenObsMLPAgent obs_dir; ignored by other agents.",
    )
    parser.add_argument(
        "--agent_model_module_path",
        default="/data/llh/navsim_workspace/navsim/scripts/evaluation/simple_token_obs_mlp.py",
        help="TokenObsMLPAgent model_module_path; ignored by other agents.",
    )
    parser.add_argument(
        "--agent_model_class_name",
        default="SimpleTokenObsMLP",
        help="TokenObsMLPAgent model_class_name; ignored by other agents.",
    )
    parser.add_argument(
        "--agent_model_config_path",
        default="/data/llh/navsim_workspace/results/token_obs_mlp_demo/config.json",
        help="TokenObsMLPAgent model_config_path; ignored by other agents.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override, e.g. --override agent.model_config_path=/path/config.json.",
    )
    parser.add_argument(
        "--draw_gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw stage 1 human/GT future trajectory as a reference.",
    )
    parser.add_argument("--duration_ms", type=int, default=120, help="GIF frame duration in milliseconds.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=2,
        help="Render every k-th 0.1s rollout frame. Use 1 for full 10Hz GIFs.",
    )
    parser.add_argument(
        "--max_agents",
        type=int,
        default=None,
        help="Optional cap on the number of closest non-ego objects drawn per frame.",
    )
    return parser.parse_args()


def normalize_stage(stage: str) -> Tuple[int, str]:
    stage_id = 1 if stage in {"1", "stage1"} else 2
    return stage_id, "stage_one" if stage_id == 1 else "stage_two"


def normalize_quality(quality: str) -> bool:
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
    raise ValueError(f"Cannot resolve sort_by='{sort_by}'. Use an exact CSV column or one of: {', '.join(supported)}")


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
        if token.startswith("extended_pdm_score") or token.startswith("average_pdm_score"):
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


def compose_cfg(
    config_name: str,
    train_test_split: str,
    agent_name: str,
    tokens: List[str],
    stage_id: int,
    extra_overrides: Sequence[str],
) -> Any:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    hydra.initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None)
    overrides = [f"train_test_split={train_test_split}", f"agent={agent_name}", *extra_overrides]
    cfg = hydra.compose(config_name=config_name, overrides=overrides)

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


def write_selected_csv(output_path: Path, rows: List[Dict[str, str]], extra_fields: Sequence[str]) -> None:
    if not rows:
        return
    preferred_fields = [
        "rank",
        "token",
        "valid",
        "score",
        "sort_column",
        "sort_value",
        "raw_pdm_score",
        "raw_no_at_fault_collisions",
        "raw_time_to_collision_within_bound",
        "rollout_gif_path",
    ]
    fields = preferred_fields + [field for field in extra_fields if field not in preferred_fields]
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compute_agent_trajectory(scene_loader: Any, agent: Any, token: str) -> Tuple[Any, Optional[Any]]:
    agent_input = scene_loader.get_agent_input_from_token(token)
    scene = None
    parameters = inspect.signature(agent.compute_trajectory).parameters
    kwargs = {"token": token} if "token" in parameters else {}
    if getattr(agent, "requires_scene", False):
        scene = scene_loader.get_scene_from_token(token)
        return agent.compute_trajectory(agent_input, scene, **kwargs), scene
    return agent.compute_trajectory(agent_input, **kwargs), scene


def rollout_case(
    token: str,
    cfg: Any,
    scene_loader: Any,
    agent: Any,
    simulator: Any,
    traffic_policy: Any,
    scorer: Any,
) -> Dict[str, Any]:
    import numpy as np

    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    metric_cache = metric_cache_loader.get_from_token(token)
    model_trajectory, scene = compute_agent_trajectory(scene_loader, agent, token)

    pred_trajectory = transform_trajectory(model_trajectory, metric_cache.ego_state)
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory,
        simulator.proposal_sampling,
        metric_cache.ego_state.time_point,
    )
    pred_states = get_trajectory_as_array(
        pred_trajectory,
        simulator.proposal_sampling,
        metric_cache.ego_state.time_point,
    )
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)
    simulated_states = simulator.simulate_proposals(trajectory_states, metric_cache.ego_state)
    ego_rollout_states = simulated_states[1]
    detections_tracks = traffic_policy.simulate_environment(ego_rollout_states, metric_cache)
    raw_score_row = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        detections_tracks,
        metric_cache.past_human_trajectory,
    )[1].iloc[0]

    return {
        "metric_cache": metric_cache,
        "scene": scene,
        "model_trajectory": model_trajectory,
        "trajectory_states": trajectory_states,
        "simulated_states": simulated_states,
        "ego_rollout_states": ego_rollout_states,
        "detections_tracks": detections_tracks,
        "raw_score_row": raw_score_row,
    }


def state_array_to_state_se2(state_array: Any) -> Any:
    from nuplan.common.actor_state.state_representation import StateSE2

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

    return StateSE2(
        float(state_array[StateIndex.X]),
        float(state_array[StateIndex.Y]),
        float(state_array[StateIndex.HEADING]),
    )


def local_box_from_global_box(box: Any, origin: Any) -> Any:
    import numpy as np
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.state_representation import StateSE2

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    center = box.center
    local_center = convert_absolute_to_relative_se2_array(
        origin,
        np.array([[center.x, center.y, center.heading]], dtype=np.float64),
    )[0]
    return OrientedBox(StateSE2(*local_center), box.length, box.width, box.height)


def plot_global_state_prefix(
    ax: Any,
    origin: Any,
    states: Sequence[Any],
    config: Dict[str, Any],
    label: str,
) -> None:
    if not states:
        return

    import numpy as np

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

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


def render_rollout_gif(
    row: Dict[str, str],
    stage_id: int,
    rollout: Dict[str, Any],
    output_dir: Path,
    draw_gt: bool,
    duration_ms: int,
    frame_stride: int,
    max_agents: Optional[int],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuplan.common.actor_state.car_footprint import CarFootprint
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    from navsim.visualization.bev import add_map_to_bev_ax, add_oriented_box_to_bev_ax
    from navsim.visualization.config import AGENT_CONFIG, BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
    from navsim.visualization.plots import configure_ax, configure_bev_ax, frame_plot_to_gif

    token = row["token"]
    metric_cache = rollout["metric_cache"]
    ego_rollout_states = rollout["ego_rollout_states"]
    detections_tracks = rollout["detections_tracks"]
    raw_score_row = rollout.get("raw_score_row")
    frame_stride = max(1, frame_stride)
    frame_indices = list(range(0, len(ego_rollout_states), frame_stride))
    if frame_indices[-1] != len(ego_rollout_states) - 1:
        frame_indices.append(len(ego_rollout_states) - 1)

    map_api = get_maps_api(
        metric_cache.map_parameters.map_root,
        metric_cache.map_parameters.map_version,
        metric_cache.map_parameters.map_name,
    )
    ego_states = [state_array_to_state_se2(state) for state in ego_rollout_states]
    gt_states = []
    if stage_id == 1 and draw_gt and metric_cache.human_trajectory is not None:
        from nuplan.common.geometry.convert import relative_to_absolute_poses

        gt_relative_states = [state_array_to_state_se2(state) for state in metric_cache.human_trajectory.poses]
        gt_states = [metric_cache.ego_state.rear_axle] + relative_to_absolute_poses(
            metric_cache.ego_state.rear_axle,
            gt_relative_states,
        )

    def plot_rollout_frame(_: Any, frame_idx: int) -> Tuple[Any, Any]:
        origin = ego_states[frame_idx]
        fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
        add_map_to_bev_ax(ax, map_api, origin)

        local_objects = []
        for tracked_object in detections_tracks[frame_idx].tracked_objects.tracked_objects:
            local_box = local_box_from_global_box(tracked_object.box, origin)
            distance = local_box.center.x**2 + local_box.center.y**2
            local_objects.append((distance, tracked_object.tracked_object_type, local_box))
        local_objects.sort(key=lambda item: item[0])
        if max_agents is not None:
            local_objects = local_objects[:max_agents]

        margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
        for _, object_type, local_box in local_objects:
            if abs(local_box.center.x) > margin_x / 2 + 10 or abs(local_box.center.y) > margin_y / 2 + 10:
                continue
            add_oriented_box_to_bev_ax(ax, local_box, AGENT_CONFIG.get(object_type, AGENT_CONFIG[TrackedObjectType.GENERIC_OBJECT]))

        ego_box = CarFootprint.build_from_rear_axle(
            rear_axle_pose=state_array_to_state_se2([0.0] * StateIndex.size()),
            vehicle_parameters=get_pacifica_parameters(),
        ).oriented_box
        add_oriented_box_to_bev_ax(ax, ego_box, AGENT_CONFIG[TrackedObjectType.EGO], add_heading=False)

        plot_global_state_prefix(
            ax,
            origin,
            ego_states[: frame_idx + 1],
            TRAJECTORY_CONFIG["agent"],
            "simulated ego",
        )
        if gt_states:
            plot_global_state_prefix(
                ax,
                origin,
                gt_states[: min(frame_idx + 1, len(gt_states))],
                TRAJECTORY_CONFIG["human"],
                "ground truth",
            )

        configure_bev_ax(ax)
        configure_ax(ax)
        ax.set_title(
            (
                f"stage {stage_id} rank {row['rank']} | t={frame_idx * 0.1:.1f}s | "
                f"csv={row.get('score', '')} | "
                f"raw={float(raw_score_row['pdm_score']):.3f} "
                f"NC={float(raw_score_row['no_at_fault_collisions']):.1f} "
                f"TTC={float(raw_score_row['time_to_collision_within_bound']):.1f}"
            ),
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        return fig, ax

    filename = (
        f"{int(row['rank']):03d}_stage{stage_id}_rollout_{safe_name(row['sort_column'])}_"
        f"{safe_name(row['sort_value'])}_{safe_name(token)}.gif"
    )
    output_path = output_dir / filename
    frame_plot_to_gif(str(output_path), plot_rollout_frame, metric_cache, frame_indices, duration=duration_ms)
    return str(output_path)


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
    extra_overrides = list(args.override)
    if args.agent == "token_obs_mlp_agent":
        extra_overrides.extend(
            [
                f"agent.obs_dir={args.agent_obs_dir}",
                f"agent.model_module_path={args.agent_model_module_path}",
                f"agent.model_class_name={args.agent_model_class_name}",
                f"agent.model_config_path={args.agent_model_config_path}",
            ]
        )
    cfg = compose_cfg(args.config_name, args.train_test_split, args.agent, tokens, stage_id, extra_overrides)

    from hydra.utils import instantiate

    agent = instantiate(cfg.agent)
    agent.initialize()
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling must be identical."
    traffic_policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
    scene_loader = build_scene_loader(cfg, agent.get_sensor_config())

    rendered_paths = []
    skipped = []
    for row in selected_rows:
        try:
            rollout = rollout_case(row["token"], cfg, scene_loader, agent, simulator, traffic_policy, scorer)
            raw_score_row = rollout["raw_score_row"]
            row["raw_pdm_score"] = f"{float(raw_score_row['pdm_score']):.8f}"
            row["raw_no_at_fault_collisions"] = f"{float(raw_score_row['no_at_fault_collisions']):.8f}"
            row["raw_time_to_collision_within_bound"] = (
                f"{float(raw_score_row['time_to_collision_within_bound']):.8f}"
            )
            gif_path = render_rollout_gif(
                row,
                stage_id,
                rollout,
                args.output_dir,
                args.draw_gt,
                args.duration_ms,
                args.frame_stride,
                args.max_agents,
            )
            row["rollout_gif_path"] = gif_path
            rendered_paths.append(gif_path)
        except Exception as exc:
            message = f"{row['token']}: {exc}"
            skipped.append(message)
            print(f"[warn] Failed to render rollout for {message}")

    write_selected_csv(
        args.output_dir / "selected_scenes.csv",
        selected_rows,
        list(fieldnames)
        + [
            "raw_pdm_score",
            "raw_no_at_fault_collisions",
            "raw_time_to_collision_within_bound",
            "rollout_gif_path",
        ],
    )
    if skipped:
        (args.output_dir / "skipped.txt").write_text("\n".join(skipped) + "\n")

    print(f"Selected {len(selected_rows)} scenes using {sort_column}.")
    print(f"Rendered {len(rendered_paths)} evaluation-aligned rollout GIFs to {args.output_dir}.")
    if skipped:
        print(f"Skipped {len(skipped)} scenes; see {args.output_dir / 'skipped.txt'}.")


if __name__ == "__main__":
    main()
