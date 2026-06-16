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
from tqdm import tqdm

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
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_gpudrive_two_stage_pdm_score"
EVALUATION_STAGES = {"all", "stage_one", "stage_two"}
INTERNAL_SCORE_COLUMNS = {"multiplicative_metrics_prod", "weighted_metrics", "weighted_metrics_array", "pdm_score"}


def _validate_evaluation_stage(evaluation_stage: str) -> None:
    if evaluation_stage not in EVALUATION_STAGES:
        raise ValueError(f"evaluation_stage must be one of {sorted(EVALUATION_STAGES)}, got {evaluation_stage}")


def _stage_enabled(evaluation_stage: str, stage: str) -> bool:
    return evaluation_stage == "all" or evaluation_stage == stage


def _show_progress(cfg: DictConfig) -> bool:
    return bool(cfg.get("show_progress", True))


def _progress(iterable: Iterable, *, enabled: bool, **kwargs):
    return tqdm(iterable, disable=not enabled, dynamic_ncols=True, **kwargs)


def _trajectory_tokens(trajectory_dir: Path, *, show_progress: bool = False, desc: str | None = None) -> set[str]:
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory does not exist: {trajectory_dir}")
    files = sorted(trajectory_dir.glob("*.npy"))
    return {
        path.stem
        for path in _progress(
            files,
            enabled=show_progress,
            desc=desc or f"scanning {trajectory_dir.name}",
            unit="file",
        )
    }


def _check_missing_trajectories(
    stage_name: str, expected_tokens: set[str], trajectory_dir: Path, *, show_progress: bool = False
) -> set[str]:
    available_tokens = _trajectory_tokens(
        trajectory_dir,
        show_progress=show_progress,
        desc=f"checking {stage_name} trajectories",
    )
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
    show_progress = _show_progress(cfg)

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
    stage_one_dir = Path(cfg.stage_one_trajectory_dir) if allowed_stage_one_tokens else None
    stage_two_dir = Path(cfg.stage_two_trajectory_dir) if allowed_stage_two_tokens else None
    metric_tokens = set(metric_cache_loader.tokens)
    pdm_results: List[pd.DataFrame] = []

    if allowed_stage_one_tokens:
        traffic_agents_policy_stage_one: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )
        tokens_to_evaluate_stage_one = sorted(
            set(scene_loader.tokens_stage_one) & metric_tokens & allowed_stage_one_tokens
        )
        stage_one_iter = _progress(
            enumerate(tokens_to_evaluate_stage_one),
            enabled=show_progress,
            total=len(tokens_to_evaluate_stage_one),
            desc=f"stage one scoring node={node_id}",
            unit="scene",
            leave=False,
        )
        for idx, token in stage_one_iter:
            if not show_progress:
                logger.info(
                    f"Processing stage one GPUDrive scenario {idx + 1} / {len(tokens_to_evaluate_stage_one)} "
                    f"in thread_id={thread_id}, node_id={node_id}"
                )
            else:
                stage_one_iter.set_postfix_str(token[:8])
            try:
                score_row_stage_one = (
                    _score_token(
                        token,
                        stage_one_dir,
                        trajectory_sampling,
                        metric_cache_loader,
                        simulator,
                        scorer,
                        traffic_agents_policy_stage_one,
                    )
                    if stage_one_dir is not None
                    else pd.DataFrame([PDMResults.get_empty_results()])
                )
            except Exception:
                logger.warning(f"----------- GPUDrive stage one trajectory failed for token {token}:")
                traceback.print_exc()
                score_row_stage_one = pd.DataFrame([PDMResults.get_empty_results()])
                score_row_stage_one["valid"] = False
                score_row_stage_one["token"] = token
            pdm_results.append(score_row_stage_one)

    if allowed_stage_two_tokens:
        traffic_agents_policy_stage_two: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )
        tokens_to_evaluate_stage_two = sorted(
            set(scene_loader.reactive_tokens_stage_two) & metric_tokens & allowed_stage_two_tokens
        )
        stage_two_iter = _progress(
            enumerate(tokens_to_evaluate_stage_two),
            enabled=show_progress,
            total=len(tokens_to_evaluate_stage_two),
            desc=f"stage two scoring node={node_id}",
            unit="scene",
            leave=False,
        )
        for idx, token in stage_two_iter:
            if not show_progress:
                logger.info(
                    f"Processing stage two GPUDrive scenario {idx + 1} / {len(tokens_to_evaluate_stage_two)} "
                    f"in thread_id={thread_id}, node_id={node_id}"
                )
            else:
                stage_two_iter.set_postfix_str(token[:8])
            try:
                score_row_stage_two = (
                    _score_token(
                        token,
                        stage_two_dir,
                        trajectory_sampling,
                        metric_cache_loader,
                        simulator,
                        scorer,
                        traffic_agents_policy_stage_two,
                    )
                    if stage_two_dir is not None
                    else pd.DataFrame([PDMResults.get_empty_results()])
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
    *,
    show_progress: bool = False,
) -> List[Dict[str, Union[List[str], DictConfig]]]:
    data_points: List[Dict[str, Union[List[str], DictConfig]]] = []
    tokens_to_evaluate = stage_one_tokens | stage_two_tokens
    for log_file, tokens_list in _progress(
        list(tokens_by_log.items()),
        enabled=show_progress,
        desc="building scoring jobs",
        unit="log",
    ):
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


def _get_tokens_list_per_log(scene_loader: SceneLoader) -> Dict[str, List[str]]:
    """Group scene tokens by log name, preserving synthetic initial tokens."""
    tokens_per_logs: Dict[str, List[str]] = {}
    for token, scene_dict_list in scene_loader.scene_frames_dicts.items():
        log_name = scene_dict_list[0]["log_name"]
        tokens_per_logs.setdefault(log_name, []).append(token)

    for token, (_, log_name) in scene_loader.synthetic_scenes.items():
        tokens_per_logs.setdefault(log_name, []).append(token)

    return tokens_per_logs


def _count_distributed_tokens(
    data_points: List[Dict[str, Union[List[str], DictConfig]]],
) -> Tuple[int, int]:
    stage_one_count = sum(len(data_point["stage_one_tokens"]) for data_point in data_points)
    stage_two_count = sum(len(data_point["stage_two_tokens"]) for data_point in data_points)
    return stage_one_count, stage_two_count


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


def _make_single_stage_pairs(
    raw_mapping: Iterable[Tuple[str, str, Iterable[Iterable[str]]]], scored_tokens: set[str], evaluation_stage: str
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for orig_token, prev_token, two_stage_pairs in raw_mapping:
        if evaluation_stage == "stage_one":
            if orig_token in scored_tokens and prev_token in scored_tokens:
                pairs.append((orig_token, prev_token))
        elif evaluation_stage == "stage_two":
            pairs.extend(
                (now_token, prev_token)
                for now_token, prev_token in (tuple(pair) for pair in two_stage_pairs)
                if now_token in scored_tokens and prev_token in scored_tokens
            )
        else:
            raise ValueError(f"single-stage pairs are only defined for stage_one/stage_two, got {evaluation_stage}")
    return pairs


def _drop_internal_score_columns(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    return pdm_score_df.drop(columns=[c for c in INTERNAL_SCORE_COLUMNS if c in pdm_score_df.columns])


def _compute_single_stage_two_frame_scores(
    pdm_score_df: pd.DataFrame,
    single_stage_pairs: List[Tuple[str, str]],
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:
    if not single_stage_pairs:
        logger.warning("No single-stage frame pairs are available; keeping original PDM scores without two-frame comfort.")
        pdm_score_df["weight"] = 1.0
        pdm_score_df["two_frame_extended_comfort"] = np.nan
        if "score" not in pdm_score_df.columns and "pdm_score" in pdm_score_df.columns:
            pdm_score_df["score"] = pdm_score_df["pdm_score"]
        return _drop_internal_score_columns(pdm_score_df)

    full_score_df = pdm_score_df.copy()
    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = 1.0
    score_df_by_token = full_score_df.set_index("token")

    updates = []
    for now_token, prev_token in single_stage_pairs:
        aggregator = SceneAggregator(
            now_frame=now_token,
            previous_frame=prev_token,
            score_df=score_df_by_token,
            proposal_sampling=proposal_sampling,
        )
        now_update = aggregator.aggregate_scores(one_stage_only=True).iloc[0]
        comfort = now_update["two_frame_extended_comfort"]
        updates.append({"token": now_token, "two_frame_extended_comfort": comfort, "weight": 1.0})
        updates.append({"token": prev_token, "two_frame_extended_comfort": comfort, "weight": 1.0})

    updates_df = pd.DataFrame(updates).drop_duplicates(subset=["token"], keep="last").set_index("token")
    score_df_by_token.update(updates_df)
    full_score_df = score_df_by_token.reset_index()

    complete_mask = full_score_df["two_frame_extended_comfort"].notna()
    if not bool(complete_mask.all()):
        missing_tokens = sorted(full_score_df.loc[~complete_mask, "token"].astype(str).tolist())
        logger.warning(
            "Missing two-frame comfort for %d single-stage tokens. "
            "These rows keep original PDM scores and will have NaN two-frame comfort. Examples: %s",
            len(missing_tokens),
            ", ".join(missing_tokens[:10]),
        )

    complete_df = full_score_df.loc[complete_mask].copy()
    incomplete_df = full_score_df.loc[~complete_mask].copy()
    output_frames: List[pd.DataFrame] = []

    if not complete_df.empty:
        output_frames.append(compute_final_scores(complete_df))
    if not incomplete_df.empty:
        if "score" not in incomplete_df.columns and "pdm_score" in incomplete_df.columns:
            incomplete_df["score"] = incomplete_df["pdm_score"]
        output_frames.append(_drop_internal_score_columns(incomplete_df))

    return pd.concat(output_frames, ignore_index=True)


def _format_metric_table(pdm_score_df: pd.DataFrame) -> str:
    rows = {
        "stage_one": "extended_pdm_score_stage_one",
        "stage_two": "extended_pdm_score_stage_two",
        "combined": "extended_pdm_score_combined",
    }
    metric_names = set()
    for column in pdm_score_df.columns:
        if column.endswith("_stage_one"):
            metric_names.add(column[: -len("_stage_one")])
        elif column.endswith("_stage_two"):
            metric_names.add(column[: -len("_stage_two")])
    metrics = ["score"] + sorted(metric_names)

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


def _score_columns(pdm_score_df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c not in INTERNAL_SCORE_COLUMNS
        )
    ]


def _add_stage_columns(pdm_score_df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
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
    return pdm_score_df[["token", "valid"] + stage1_cols + stage2_cols + ["score"]]


def _append_single_stage_average(pdm_score_df: pd.DataFrame, evaluation_stage: str) -> pd.DataFrame:
    stage_suffix = evaluation_stage
    stage_cols = [col for col in pdm_score_df.columns if col.endswith(f"_{stage_suffix}")]
    average_cols = stage_cols + ["score"]
    average_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    average_row["token"] = f"average_pdm_score_{stage_suffix}"
    average_row["valid"] = bool(pdm_score_df["valid"].all())
    for col in average_cols:
        average_row[col] = pd.to_numeric(pdm_score_df[col], errors="coerce").mean(skipna=True)
    return pd.concat([pdm_score_df, pd.DataFrame([average_row])], ignore_index=True)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)
    evaluation_stage = str(cfg.evaluation_stage)
    show_progress = _show_progress(cfg)
    _validate_evaluation_stage(evaluation_stage)

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

    stage_one_expected_tokens = set(scene_loader.tokens_stage_one) if _stage_enabled(evaluation_stage, "stage_one") else set()
    stage_two_expected_tokens = (
        set(scene_loader.reactive_tokens_stage_two) if _stage_enabled(evaluation_stage, "stage_two") else set()
    )
    stage_one_tokens = stage_one_expected_tokens & metric_tokens
    stage_two_tokens = stage_two_expected_tokens & metric_tokens
    scene_tokens = stage_one_expected_tokens | stage_two_expected_tokens

    stage_one_dir = Path(cfg.stage_one_trajectory_dir) if _stage_enabled(evaluation_stage, "stage_one") else None
    stage_two_dir = Path(cfg.stage_two_trajectory_dir) if _stage_enabled(evaluation_stage, "stage_two") else None
    if cfg.strict_trajectory_files:
        stage_one_available_tokens = (
            _check_missing_trajectories("stage one", stage_one_tokens, stage_one_dir, show_progress=show_progress)
            if stage_one_dir is not None
            else set()
        )
        stage_two_available_tokens = (
            _check_missing_trajectories("stage two", stage_two_tokens, stage_two_dir, show_progress=show_progress)
            if stage_two_dir is not None
            else set()
        )
    else:
        stage_one_available_tokens = (
            _trajectory_tokens(
                stage_one_dir,
                show_progress=show_progress,
                desc="scanning available stage one trajectories",
            )
            if stage_one_dir is not None
            else set()
        )
        stage_two_available_tokens = (
            _trajectory_tokens(
                stage_two_dir,
                show_progress=show_progress,
                desc="scanning available stage two trajectories",
            )
            if stage_two_dir is not None
            else set()
        )
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
        f"Starting GPUDrive {evaluation_stage} PDM scoring of "
        f"{len(stage_one_eval_tokens)} stage one and {len(stage_two_eval_tokens)} stage two scenarios..."
    )
    data_points = _build_data_points(
        cfg,
        stage_one_eval_tokens,
        stage_two_eval_tokens,
        _get_tokens_list_per_log(scene_loader),
        show_progress=show_progress,
    )
    if not data_points:
        raise RuntimeError(
            f"No GPUDrive {evaluation_stage} PDM scoring jobs were built. "
            "Check evaluation_stage, trajectory directories, and scene filter tokens."
        )
    distributed_stage_one_count, distributed_stage_two_count = _count_distributed_tokens(data_points)
    logger.info(
        f"Built {len(data_points)} GPUDrive PDM scoring jobs with "
        f"{distributed_stage_one_count} stage one and {distributed_stage_two_count} stage two tokens."
    )
    if stage_one_eval_tokens and distributed_stage_one_count == 0:
        raise RuntimeError("Stage one trajectories are available, but no stage one tokens were distributed.")
    if stage_two_eval_tokens and distributed_stage_two_count == 0:
        raise RuntimeError(
            "Stage two trajectories are available, but no stage two tokens were distributed. "
            "Check synthetic scene token grouping."
        )
    score_rows: List[pd.DataFrame] = worker_map(worker, run_gpudrive_two_stage_pdm_score, data_points)
    pdm_score_df = pd.concat(score_rows, ignore_index=True)

    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    pseudo_closed_loop_valid = False
    if evaluation_stage == "all":
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
    else:
        scored_tokens = set(pdm_score_df["token"])
        single_stage_pairs = _make_single_stage_pairs(
            cfg.train_test_split.reactive_all_mapping, scored_tokens, evaluation_stage
        )
        pdm_score_df = _compute_single_stage_two_frame_scores(
            pdm_score_df,
            single_stage_pairs,
            instantiate(cfg.simulator.proposal_sampling),
        )

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list() if num_failed_scenarios > 0 else []

    score_cols = _score_columns(pdm_score_df)

    if all_mappings:
        pcl_group_score, pcl_stage1_score, pcl_stage2_score = calculate_individual_mapping_scores(
            pdm_score_df[score_cols + ["token", "weight"]], all_mappings
        )
    else:
        empty_scores = pd.Series({col: np.nan for col in score_cols})
        pcl_group_score = empty_scores
        pcl_stage1_score = empty_scores
        pcl_stage2_score = empty_scores

    pdm_score_df = _add_stage_columns(pdm_score_df, score_cols)

    summary_rows = []
    if evaluation_stage == "all":
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
    else:
        pdm_score_df = _append_single_stage_average(pdm_score_df, evaluation_stage)

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
            Final pdm score of valid results: {pd.to_numeric(pdm_score_df["score"], errors="coerce").mean(skipna=True)}.
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
