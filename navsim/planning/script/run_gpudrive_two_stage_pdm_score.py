from __future__ import annotations

import logging
import os
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.common.dataclasses import PDMResults, SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.run_pdm_score import (
    calculate_individual_mapping_scores,
    compute_final_scores,
    create_scene_aggregators,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_gpudrive_two_stage_pdm_score"


def _trajectory_tokens(trajectory_dir: Path) -> set[str]:
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory does not exist: {trajectory_dir}")
    return {path.stem for path in trajectory_dir.glob("*.npy")}


def _check_missing_trajectories(stage_name: str, expected_tokens: set[str], trajectory_dir: Path) -> set[str]:
    available_tokens = _trajectory_tokens(trajectory_dir)
    missing_tokens = sorted(expected_tokens - available_tokens)
    if missing_tokens:
        examples = ", ".join(missing_tokens[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_tokens)} {stage_name} trajectory files in {trajectory_dir}. "
            f"Examples: {examples}"
        )
    return available_tokens


def _load_trajectory(token: str, trajectory_dir: Path, trajectory_sampling: TrajectorySampling) -> Trajectory:
    path = trajectory_dir / f"{token}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory not found for token {token}: {path}")

    poses = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    expected_shape = (trajectory_sampling.num_poses, 3)
    if poses.shape != expected_shape:
        raise ValueError(f"Trajectory {path} has shape {poses.shape}; expected {expected_shape}")
    if not np.isfinite(poses).all():
        raise ValueError(f"Trajectory contains non-finite values: {path}")
    return Trajectory(poses=poses, trajectory_sampling=trajectory_sampling)


def _score_token(
    token: str,
    trajectory_dir: Path,
    trajectory_sampling: TrajectorySampling,
    metric_cache_loader: MetricCacheLoader,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
) -> pd.DataFrame:
    metric_cache = metric_cache_loader.get_from_token(token)
    trajectory = _load_trajectory(token, trajectory_dir, trajectory_sampling)

    score_row, ego_simulated_states = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )
    score_row["valid"] = True
    score_row["log_name"] = metric_cache.log_name
    score_row["frame_type"] = metric_cache.scene_type
    score_row["start_time"] = metric_cache.timepoint.time_s

    end_pose = StateSE2(
        x=trajectory.poses[-1, 0],
        y=trajectory.poses[-1, 1],
        heading=trajectory.poses[-1, 2],
    )
    absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
    score_row["endpoint_x"] = absolute_endpoint.x
    score_row["endpoint_y"] = absolute_endpoint.y
    score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
    score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
    score_row["ego_simulated_states"] = [ego_simulated_states]
    score_row["token"] = token
    return score_row


def run_gpudrive_two_stage_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting GPUDrive two-stage PDM worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    allowed_stage_one_tokens = {t for a in args for t in a["stage_one_tokens"]}
    allowed_stage_two_tokens = {t for a in args for t in a["stage_two_tokens"]}
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    trajectory_sampling: TrajectorySampling = instantiate(cfg.trajectory_sampling)
    stage_one_dir = Path(cfg.stage_one_trajectory_dir)
    stage_two_dir = Path(cfg.stage_two_trajectory_dir)
    metric_tokens = set(metric_cache_loader.tokens)
    pdm_results: List[pd.DataFrame] = []

    traffic_agents_policy_stage_one: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )
    tokens_to_evaluate_stage_one = sorted(set(scene_loader.tokens_stage_one) & metric_tokens & allowed_stage_one_tokens)
    for idx, token in enumerate(tokens_to_evaluate_stage_one):
        logger.info(
            f"Processing stage one GPUDrive scenario {idx + 1} / {len(tokens_to_evaluate_stage_one)} "
            f"in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            score_row_stage_one = _score_token(
                token,
                stage_one_dir,
                trajectory_sampling,
                metric_cache_loader,
                simulator,
                scorer,
                traffic_agents_policy_stage_one,
            )
        except Exception:
            logger.warning(f"----------- GPUDrive stage one trajectory failed for token {token}:")
            traceback.print_exc()
            score_row_stage_one = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_one["valid"] = False
            score_row_stage_one["token"] = token
        pdm_results.append(score_row_stage_one)

    traffic_agents_policy_stage_two: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )
    tokens_to_evaluate_stage_two = sorted(
        set(scene_loader.reactive_tokens_stage_two) & metric_tokens & allowed_stage_two_tokens
    )
    for idx, token in enumerate(tokens_to_evaluate_stage_two):
        logger.info(
            f"Processing stage two GPUDrive scenario {idx + 1} / {len(tokens_to_evaluate_stage_two)} "
            f"in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            score_row_stage_two = _score_token(
                token,
                stage_two_dir,
                trajectory_sampling,
                metric_cache_loader,
                simulator,
                scorer,
                traffic_agents_policy_stage_two,
            )
        except Exception:
            logger.warning(f"----------- GPUDrive stage two trajectory failed for token {token}:")
            traceback.print_exc()
            score_row_stage_two = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_two["valid"] = False
            score_row_stage_two["token"] = token
        pdm_results.append(score_row_stage_two)

    return pdm_results


def _build_data_points(
    cfg: DictConfig,
    stage_one_tokens: set[str],
    stage_two_tokens: set[str],
    tokens_by_log: Dict[str, List[str]],
) -> List[Dict[str, Union[List[str], DictConfig]]]:
    data_points: List[Dict[str, Union[List[str], DictConfig]]] = []
    tokens_to_evaluate = stage_one_tokens | stage_two_tokens
    for log_file, tokens_list in tokens_by_log.items():
        selected_tokens = sorted(set(tokens_list) & tokens_to_evaluate)
        if selected_tokens:
            data_points.append(
                {
                    "cfg": cfg,
                    "log_file": log_file,
                    "tokens": selected_tokens,
                    "stage_one_tokens": sorted(set(selected_tokens) & stage_one_tokens),
                    "stage_two_tokens": sorted(set(selected_tokens) & stage_two_tokens),
                }
            )
    return data_points


def _make_mapping(
    raw_mapping: Iterable[Tuple[str, str, Iterable[Iterable[str]]]], scored_tokens: set[str]
) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for orig_token, prev_token, two_stage_pairs in raw_mapping:
        pairs = [tuple(pair) for pair in two_stage_pairs]
        pair_tokens = {token for pair in pairs for token in pair}
        if orig_token in scored_tokens and prev_token in scored_tokens and pair_tokens <= scored_tokens:
            all_mappings[(orig_token, prev_token)] = pairs
    return all_mappings


def _format_metric_table(pdm_score_df: pd.DataFrame) -> str:
    rows = {
        "stage_one": "extended_pdm_score_stage_one",
        "stage_two": "extended_pdm_score_stage_two",
        "combined": "extended_pdm_score_combined",
    }
    metrics = [
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "history_comfort",
        "two_frame_extended_comfort",
    ]

    table = pd.DataFrame(index=rows.keys(), columns=metrics, dtype=float)

    def _as_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def _mean_column(column: str, mask: pd.Series | None = None) -> float:
        if column not in pdm_score_df.columns:
            return np.nan
        values = pd.to_numeric(pdm_score_df.loc[mask, column] if mask is not None else pdm_score_df[column], errors="coerce")
        return float(values.mean(skipna=True))

    for row_name, token in rows.items():
        row = pdm_score_df[pdm_score_df["token"] == token]
        stage_col_suffix = row_name if row_name in {"stage_one", "stage_two"} else None
        if stage_col_suffix == "stage_one":
            fallback_mask = pdm_score_df.filter(like="_stage_one").notna().any(axis=1)
        elif stage_col_suffix == "stage_two":
            fallback_mask = pdm_score_df.filter(like="_stage_two").notna().any(axis=1)
        else:
            fallback_mask = ~pdm_score_df["token"].astype(str).str.startswith("extended_pdm_score_")

        if row.empty:
            row = pd.Series(dtype=object)
        else:
            row = row.iloc[0]
        for metric in metrics:
            if metric == "score":
                value = _as_float(row.get(metric, np.nan))
                table.loc[row_name, metric] = value if np.isfinite(value) else _mean_column(metric, fallback_mask)
            else:
                stage_col = f"{metric}_{stage_col_suffix}" if stage_col_suffix else None
                if stage_col and stage_col in pdm_score_df.columns:
                    value = _as_float(row.get(stage_col, np.nan))
                    table.loc[row_name, metric] = value if np.isfinite(value) else _mean_column(stage_col, fallback_mask)
                else:
                    stage_one_col = f"{metric}_stage_one"
                    stage_two_col = f"{metric}_stage_two"
                    stage_one_value = _as_float(row.get(stage_one_col, np.nan))
                    stage_two_value = _as_float(row.get(stage_two_col, np.nan))
                    if np.isfinite(stage_one_value) or np.isfinite(stage_two_value):
                        table.loc[row_name, metric] = np.nanmean([stage_one_value, stage_two_value])
                    else:
                        fallback_values = [
                            _mean_column(stage_one_col, fallback_mask),
                            _mean_column(stage_two_col, fallback_mask),
                        ]
                        finite_values = [value for value in fallback_values if np.isfinite(value)]
                        table.loc[row_name, metric] = float(np.mean(finite_values)) if finite_values else np.nan

    return table.to_string(float_format=lambda value: f"{value:.4f}" if np.isfinite(value) else "nan")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)

    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    metric_tokens = set(metric_cache_loader.tokens)

    stage_one_tokens = set(scene_loader.tokens_stage_one) & metric_tokens
    stage_two_tokens = set(scene_loader.reactive_tokens_stage_two) & metric_tokens
    scene_tokens = set(scene_loader.tokens)

    stage_one_dir = Path(cfg.stage_one_trajectory_dir)
    stage_two_dir = Path(cfg.stage_two_trajectory_dir)
    if cfg.strict_trajectory_files:
        stage_one_available_tokens = _check_missing_trajectories("stage one", stage_one_tokens, stage_one_dir)
        stage_two_available_tokens = _check_missing_trajectories("stage two", stage_two_tokens, stage_two_dir)
    else:
        stage_one_available_tokens = _trajectory_tokens(stage_one_dir)
        stage_two_available_tokens = _trajectory_tokens(stage_two_dir)
        logger.warning(
            "strict_trajectory_files=false: evaluating only tokens with available GPUDrive trajectories. "
            f"Missing stage one: {len(stage_one_tokens - stage_one_available_tokens)}, "
            f"missing stage two: {len(stage_two_tokens - stage_two_available_tokens)}"
        )

    stage_one_eval_tokens = stage_one_tokens & stage_one_available_tokens
    stage_two_eval_tokens = stage_two_tokens & stage_two_available_tokens
    tokens_to_evaluate = stage_one_eval_tokens | stage_two_eval_tokens

    num_missing_metric_cache_tokens = len(scene_tokens - metric_tokens)
    num_unused_metric_cache_tokens = len(metric_tokens - scene_tokens)
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")

    logger.info(
        "Starting GPUDrive two-stage PDM scoring of "
        f"{len(stage_one_eval_tokens)} stage one and {len(stage_two_eval_tokens)} stage two scenarios..."
    )
    data_points = _build_data_points(
        cfg, stage_one_eval_tokens, stage_two_eval_tokens, scene_loader.get_tokens_list_per_log()
    )
    score_rows: List[pd.DataFrame] = worker_map(worker, run_gpudrive_two_stage_pdm_score, data_points)
    pdm_score_df = pd.concat(score_rows, ignore_index=True)

    scored_tokens = set(pdm_score_df["token"])
    all_mappings = _make_mapping(cfg.train_test_split.reactive_all_mapping, scored_tokens)
    try:
        if not all_mappings:
            raise ValueError("No complete two-stage mappings are available for the evaluated tokens.")
        pdm_score_df = create_scene_aggregators(
            all_mappings, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
        )
        pdm_score_df = compute_final_scores(pdm_score_df)
        pseudo_closed_loop_valid = True
    except Exception:
        logger.warning("----------- Failed to calculate pseudo closed-loop weights or comfort:")
        traceback.print_exc()
        pdm_score_df["weight"] = 1.0
        pdm_score_df["two_frame_extended_comfort"] = np.nan
        if "score" not in pdm_score_df.columns and "pdm_score" in pdm_score_df.columns:
            pdm_score_df["score"] = pdm_score_df["pdm_score"]
        pseudo_closed_loop_valid = False

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list() if num_failed_scenarios > 0 else []

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    if all_mappings:
        pcl_group_score, pcl_stage1_score, pcl_stage2_score = calculate_individual_mapping_scores(
            pdm_score_df[score_cols + ["token", "weight"]], all_mappings
        )
    else:
        empty_scores = pd.Series({col: np.nan for col in score_cols})
        pcl_group_score = empty_scores
        pcl_stage1_score = empty_scores
        pcl_stage2_score = empty_scores

    for col in score_cols:
        stage_one_mask = pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
        stage_two_mask = pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC
        pdm_score_df.loc[stage_one_mask, f"{col}_stage_one"] = pdm_score_df.loc[stage_one_mask, col]
        pdm_score_df.loc[stage_two_mask, f"{col}_stage_two"] = pdm_score_df.loc[stage_two_mask, col]

    pdm_score_df.drop(columns=score_cols, inplace=True)
    for stage_score_col in ["score_stage_one", "score_stage_two"]:
        if stage_score_col not in pdm_score_df.columns:
            pdm_score_df[stage_score_col] = np.nan
    pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(pdm_score_df["score_stage_two"])
    pdm_score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)

    stage1_cols = [f"{col}_stage_one" for col in score_cols if col != "score"]
    stage2_cols = [f"{col}_stage_two" for col in score_cols if col != "score"]
    final_score_cols = stage1_cols + stage2_cols + ["score"]
    pdm_score_df = pdm_score_df[["token", "valid"] + final_score_cols]

    summary_rows = []
    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = pseudo_closed_loop_valid
    stage1_row["score"] = pcl_stage1_score.get("score", np.nan)
    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            stage1_row[f"{col}_stage_one"] = pcl_stage1_score[col]
    summary_rows.append(stage1_row)

    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = pseudo_closed_loop_valid
    stage2_row["score"] = pcl_stage2_score.get("score", np.nan)
    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            stage2_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(stage2_row)

    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = pseudo_closed_loop_valid
    combined_row["score"] = pcl_group_score["score"]
    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_one"] = pcl_stage1_score[col]
    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(combined_row)

    pdm_score_df = pd.concat([pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True)

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    result_path = save_path / f"{timestamp}.csv"
    pdm_score_df.to_csv(result_path)

    metric_table = _format_metric_table(pdm_score_df)
    logger.info(
        f"""
        Finished running GPUDrive two-stage evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final extended pdm score of valid results: {pdm_score_df[pdm_score_df["token"] == "extended_pdm_score_combined"]["score"].iloc[0]}.
            Results are stored in: {result_path}.

        Metric summary:
        {metric_table}
        """
    )
    print(f"\nMetric summary:\n{metric_table}\n")

    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )


if __name__ == "__main__":
    main()
