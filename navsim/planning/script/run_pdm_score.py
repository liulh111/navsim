import logging
import os
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

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

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import BoundingBoxIndex, SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)


def _extract_whitebox_features(scene, token: str, log: logging.Logger) -> bool:
    """
    Extract and log a *probe* of the Waymo whitebox features from a navsim Scene.
    Fires once per worker (caller flips a flag after the first call).

    Confirmed from run output:
      - ego_status.ego_pose is a numpy array [x, y, heading], NOT StateSE2
      - ego_status.ego_velocity is a numpy array (2- or 3-element)
    """
    frame_idx = scene.scene_metadata.num_history_frames - 1
    ego_status = scene.frames[frame_idx].ego_status
    annotations = scene.frames[frame_idx].annotations

    # ego_pose is a raw ndarray [x, y, heading] — NOT a StateSE2 object
    ego_pose_arr = ego_status.ego_pose          # shape (3,)
    ego_x       = float(ego_pose_arr[0])
    ego_y       = float(ego_pose_arr[1])
    ego_heading = float(ego_pose_arr[2])

    # ── 1. ego-LocalEgoState (1×6) ───────────────────────────────────────────
    speed = float(np.linalg.norm(ego_status.ego_velocity))

    try:
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        params = get_pacifica_parameters()
        vehicle_length = float(params.length)
        vehicle_width  = float(params.width)
    except Exception:
        vehicle_length, vehicle_width = 5.176, 2.297  # confirmed constants from run

    trajectory = scene.get_future_trajectory(
        num_trajectory_frames=scene.scene_metadata.num_future_frames
    )
    rel_goal_x = float(trajectory.poses[-1, 0])
    rel_goal_y = float(trajectory.poses[-1, 1])
    is_collided = 0  # not available from raw scene data

    ego_state = np.array([[speed, vehicle_length, vehicle_width,
                           rel_goal_x, rel_goal_y, is_collided]])  # (1, 6)

    # ── 2. partner_state (2×6 probe) ─────────────────────────────────────────
    N = annotations.boxes.shape[0]
    probe_n = min(2, N)
    if probe_n > 0:
        partner_state_probe = np.stack([
            np.linalg.norm(annotations.velocity_3d[:probe_n, :2], axis=1),
            annotations.boxes[:probe_n, BoundingBoxIndex.X],
            annotations.boxes[:probe_n, BoundingBoxIndex.Y],
            annotations.boxes[:probe_n, BoundingBoxIndex.HEADING],
            annotations.boxes[:probe_n, BoundingBoxIndex.LENGTH],
            annotations.boxes[:probe_n, BoundingBoxIndex.WIDTH],
        ], axis=1)  # (probe_n, 6)
    else:
        partner_state_probe = np.zeros((0, 6), dtype=np.float32)

    # ── 3. LocalRoadState (2×13 probe) ───────────────────────────────────────
    road_rows = []
    road_debug_lines = []   # per-layer diagnostics, always printed
    try:
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import Point2D

        query_point = Point2D(ego_x, ego_y)

        # SPEED_BUMP may be unavailable in some map versions — query separately
        # so a missing layer doesn't suppress results from the others
        layers_always = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.STOP_LINE,
        ]
        layers_optional = [SemanticMapLayer.SPEED_BUMP]
        layer_onehot = {
            SemanticMapLayer.LANE:           3,
            SemanticMapLayer.LANE_CONNECTOR: 3,
            SemanticMapLayer.ROADBLOCK:      2,
            SemanticMapLayer.CROSSWALK:      4,
            SemanticMapLayer.SPEED_BUMP:     5,
            SemanticMapLayer.STOP_LINE:      6,
        }
        map_objects = scene.map_api.get_proximal_map_objects(
            point=query_point, radius=50.0, layers=layers_always
        )
        try:
            opt_objs = scene.map_api.get_proximal_map_objects(
                point=query_point, radius=50.0, layers=layers_optional
            )
            map_objects.update(opt_objs)
        except Exception as _opt_e:
            road_debug_lines.append(f"  optional layers skipped: {_opt_e}")

        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)

        for layer, objs in map_objects.items():
            layer_name = layer.name
            road_debug_lines.append(
                f"  layer {layer_name}: {len(objs)} object(s) found"
            )
            for obj_idx, obj in enumerate(objs[:3]):  # inspect up to 3 per layer
                obj_attrs = [a for a in ("baseline_path", "polygon", "exterior")
                             if hasattr(obj, a)]
                road_debug_lines.append(
                    f"    obj[{obj_idx}] type={type(obj).__name__}  "
                    f"has_attrs={obj_attrs}"
                )
                if len(road_rows) >= 2:
                    continue  # keep logging even after we have 2 rows

                # ── try baseline_path first (Lane / LaneConnector) ──
                geom_src = None
                seg_width = 0.0
                try:
                    ls = obj.baseline_path.linestring
                    coords = np.array(ls.coords)
                    geom_src = "baseline_path"
                    seg_len = float(ls.length) / max(len(coords) - 1, 1)
                    try:
                        seg_width = float(obj.width_at_point(
                            ls.interpolate(0.5, normalized=True)
                        ))
                    except Exception as _we:
                        road_debug_lines.append(
                            f"      width_at_point failed: {type(_we).__name__}({_we})"
                        )
                except Exception as _bp_e:
                    road_debug_lines.append(
                        f"      baseline_path failed: {type(_bp_e).__name__}({_bp_e})"
                    )
                    # ── fallback: polygon exterior (RoadBlock / CrossWalk / StopLine) ──
                    try:
                        poly = obj.polygon
                        ext = np.array(poly.exterior.coords)  # (M, 2)
                        # use polygon exterior as coords; seg_len = perimeter / num_segments
                        coords = ext
                        seg_len = float(poly.length) / max(len(ext) - 1, 1)
                        # approximate width from bounding-box minor axis
                        bb_w = float(poly.minimum_rotated_rectangle.exterior.length) / 4
                        seg_width = bb_w
                        geom_src = "polygon"
                    except Exception as _pg_e:
                        road_debug_lines.append(
                            f"      polygon fallback also failed: "
                            f"{type(_pg_e).__name__}({_pg_e})"
                        )
                        continue

                # ── global → ego-local transform ──
                dx = coords[:, 0] - ego_x
                dy = coords[:, 1] - ego_y
                lx =  dx * cos_h - dy * sin_h
                ly =  dx * sin_h + dy * cos_h
                cx, cy = float(np.mean(lx)), float(np.mean(ly))
                seg_height = 0.0  # 2-D map
                orientation = (
                    float(np.arctan2(coords[1, 1] - coords[0, 1],
                                      coords[1, 0] - coords[0, 0])) - ego_heading
                    if len(coords) >= 2 else 0.0
                )
                type_onehot = [0] * 7
                type_onehot[layer_onehot.get(layer, 0)] = 1
                road_rows.append(
                    [cx, cy, seg_len, seg_width, seg_height, orientation]
                    + type_onehot
                )
                road_debug_lines.append(
                    f"      -> SUCCESS via {geom_src}: "
                    f"cx={cx:.3f} cy={cy:.3f} seg_len={seg_len:.3f} "
                    f"seg_w={seg_width:.3f} orient={orientation:.4f} "
                    f"type={type_onehot}"
                )

    except Exception as _me:
        road_debug_lines.append(f"  OUTER ERROR: {type(_me).__name__}({_me})")

    road_state_probe = (
        np.array(road_rows[:2], dtype=np.float32)
        if road_rows else
        np.zeros((0, 13), dtype=np.float32)
    )

    # ── Log results ───────────────────────────────────────────────────────────
    sep = "=" * 64
    log.info(f"\n{sep}")
    log.info(f"WHITEBOX FEATURE PROBE  token={token}")
    log.info(f"{sep}")
    log.info(
        f"[ego_state 1x6]  speed={ego_state[0,0]:.4f}  "
        f"vehicle_length={ego_state[0,1]:.4f}  vehicle_width={ego_state[0,2]:.4f}  "
        f"rel_goal_x={ego_state[0,3]:.4f}  rel_goal_y={ego_state[0,4]:.4f}  "
        f"is_collided={int(ego_state[0,5])}"
    )
    log.info(
        f"  -> ego_velocity raw shape={np.array(ego_status.ego_velocity).shape}  "
        f"values={np.array(ego_status.ego_velocity).tolist()}"
    )
    log.info(
        f"  -> ego_pose raw shape={np.array(ego_status.ego_pose).shape}  "
        f"[x={ego_x:.3f}, y={ego_y:.3f}, heading={ego_heading:.4f}]  type={type(ego_status.ego_pose).__name__}"
    )
    log.info(
        f"  -> num_future_frames={scene.scene_metadata.num_future_frames}  "
        f"trajectory.poses shape={trajectory.poses.shape}"
    )
    log.info(f"\n[partner_state 2x6 probe]  total agents in frame: {N}")
    log.info(f"  cols: [speed, rel_pos_x, rel_pos_y, orientation(heading), vehicle_length, vehicle_width]")
    log.info(f"  -> annotations.boxes.shape={annotations.boxes.shape}  "
             f"velocity_3d.shape={annotations.velocity_3d.shape}")
    for i, row in enumerate(partner_state_probe):
        log.info(f"  agent[{i}]: speed={row[0]:.4f}  x={row[1]:.4f}  y={row[2]:.4f}  "
                 f"heading={row[3]:.4f}  length={row[4]:.4f}  width={row[5]:.4f}")
    log.info(
        f"\n[road_state 2x13 probe]  "
        f"(radius=50m, layers: LANE/LANE_CONNECTOR/ROADBLOCK/CROSSWALK/SPEED_BUMP/STOP_LINE)"
    )
    log.info(f"  cols: [x, y, seg_len, seg_width, seg_height, orientation, "
             f"none, RoadLine, RoadEdge, RoadLane, CrossWalk, SpeedBump, StopSign]")
    for dbg in road_debug_lines:
        log.info(dbg)
    for i, row in enumerate(road_state_probe):
        log.info(f"  road[{i}]: x={row[0]:.4f}  y={row[1]:.4f}  "
                 f"seg_len={row[2]:.4f}  seg_w={row[3]:.4f}  seg_h={row[4]:.4f}  "
                 f"orient={row[5]:.4f}  type={[int(v) for v in row[6:]]}")
    log.info(sep + "\n")
    return True


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

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
        sensor_config=agent.get_sensor_config(),
    )

    pdm_results: List[pd.DataFrame] = []

    # first stage

    traffic_agents_policy_stage_one: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )

    scene_loader_tokens_stage_one = scene_loader.tokens_stage_one

    tokens_to_evaluate_stage_one = list(set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens))
    _whitebox_probe_done = False  # fire the feature probe exactly once per worker
    for idx, (token) in enumerate(tokens_to_evaluate_stage_one):
        logger.info(
            f"Processing stage one reactive scenario {idx + 1} / {len(tokens_to_evaluate_stage_one)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)  # moved here for whitebox probe
            if not _whitebox_probe_done:
                try:
                    _whitebox_probe_done = _extract_whitebox_features(scene, token, logger)
                except Exception as _probe_err:
                    logger.warning(f"Whitebox probe failed for token {token}: {_probe_err}")
                    _whitebox_probe_done = True  # don't retry on every token
            if agent.requires_scene:
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            score_row_stage_one, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy_stage_one,
            )
            score_row_stage_one["valid"] = True
            score_row_stage_one["log_name"] = metric_cache.log_name
            score_row_stage_one["frame_type"] = metric_cache.scene_type
            score_row_stage_one["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row_stage_one["endpoint_x"] = absolute_endpoint.x
            score_row_stage_one["endpoint_y"] = absolute_endpoint.y
            score_row_stage_one["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row_stage_one["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row_stage_one["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row_stage_one = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_one["valid"] = False
        score_row_stage_one["token"] = token

        pdm_results.append(score_row_stage_one)

    # second stage

    traffic_agents_policy_stage_two: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )
    scene_loader_tokens_stage_two = scene_loader.reactive_tokens_stage_two

    tokens_to_evaluate_stage_two = list(set(scene_loader_tokens_stage_two) & set(metric_cache_loader.tokens))
    for idx, (token) in enumerate(tokens_to_evaluate_stage_two):
        logger.info(
            f"Processing stage two reactive scenario {idx + 1} / {len(tokens_to_evaluate_stage_two)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)  # symmetric with stage one
            if agent.requires_scene:
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            score_row_stage_two, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy_stage_two,
            )
            score_row_stage_two["valid"] = True
            score_row_stage_two["log_name"] = metric_cache.log_name
            score_row_stage_two["frame_type"] = metric_cache.scene_type
            score_row_stage_two["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row_stage_two["endpoint_x"] = absolute_endpoint.x
            score_row_stage_two["endpoint_y"] = absolute_endpoint.y
            score_row_stage_two["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row_stage_two["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row_stage_two["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row_stage_two = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_two["valid"] = False
        score_row_stage_two["token"] = token

        pdm_results.append(score_row_stage_two)

    return pdm_results


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final scores after injecting the two-frame extended comfort score
    into the weighted metrics.

    :param pdm_score_df: Original PDM score DataFrame with all tokens
    :return: DataFrame with final score computed
    """
    df = pdm_score_df.reset_index()

    assert (
        not df["two_frame_extended_comfort"].isna().any()
    ), "Found NaN in 'two_frame_extended_comfort'. Please check aggregator completeness."

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())  # shape: (N, M)
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())  # shape: (N, M)

    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    weighted_metrics[:, two_frame_idx] = two_frame_scores

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)

    assert np.all(total_weight > 0), "Found total_weight == 0 during score computation."

    weighted_metric_scores = weighted_sum / total_weight
    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores

    df.drop(columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"], inplace=True)

    return df


def calculate_weighted_average_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the weighted average for all score columns in the dataframe.
    Automatically normalizes weights if they do not sum to 1.
    """
    score_cols = [c for c in df.columns if c not in {"weight", "token"}]

    if df.empty:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weights = df["weight"].to_numpy()
    scores = df[score_cols].to_numpy()
    total_weight = weights.sum()

    if total_weight == 0:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weighted_avg = (scores * weights[:, None]).sum(axis=0) / total_weight
    return pd.Series(weighted_avg, index=score_cols)


def calculate_individual_mapping_scores(
    pdm_score_df: pd.DataFrame, all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]]
) -> pd.Series:
    """
    Args:
        pdm_score_df: DataFrame containing scores for all tokens
        all_mappings: Two-stage mapping dictionary
    Returns:
        Average score calculated using group multiplication
    """
    all_group_scores = []
    stage1_group_scores = []
    stage2_group_scores = []

    for (orig_token, prev_token), second_stage_pairs in all_mappings.items():

        first_tokens = [pair[0] for pair in second_stage_pairs if len(pair) > 0]
        second_tokens = [pair[1] for pair in second_stage_pairs if len(pair) > 1]

        group1_stage1_df = pdm_score_df[pdm_score_df["token"] == orig_token]
        group1_stage2_df = pdm_score_df[pdm_score_df["token"].isin(first_tokens)]

        group2_stage1_df = pdm_score_df[pdm_score_df["token"] == prev_token]
        group2_stage2_df = pdm_score_df[pdm_score_df["token"].isin(second_tokens)]

        group1_stage1_scores = calculate_weighted_average_score(group1_stage1_df)
        group1_stage2_scores = calculate_weighted_average_score(group1_stage2_df)

        group2_stage1_scores = calculate_weighted_average_score(group2_stage1_df)
        group2_stage2_scores = calculate_weighted_average_score(group2_stage2_df)

        stage1_group_scores.append(group1_stage1_scores)
        stage1_group_scores.append(group2_stage1_scores)

        stage2_group_scores.append(group1_stage2_scores)
        stage2_group_scores.append(group2_stage2_scores)

        group1_scores = group1_stage1_scores * group1_stage2_scores

        group2_scores = group2_stage1_scores * group2_stage2_scores

        avg_scores = (group1_scores + group2_scores) / 2
        all_group_scores.append(avg_scores)

    return (
        pd.DataFrame(all_group_scores).mean(),
        pd.DataFrame(stage1_group_scores).mean(),
        pd.DataFrame(stage2_group_scores).mean(),
    )


def create_scene_aggregators(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []

    for (now_frame, previous_frame), second_stage in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            second_stage=second_stage,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores()

        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])

    return full_score_df


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.concat(score_rows)

    try:
        raw_mapping = cfg.train_test_split.reactive_all_mapping
        all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

        for orig_token, prev_token, two_stage_pairs in raw_mapping:
            if prev_token in set(scene_loader.tokens) or orig_token in set(scene_loader.tokens):
                all_mappings[(orig_token, prev_token)] = [tuple(pair) for pair in two_stage_pairs]

        pdm_score_df = create_scene_aggregators(
            all_mappings, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
        )
        pdm_score_df = compute_final_scores(pdm_score_df)
        pseudo_closed_loop_valid = True

    except Exception:
        logger.warning("----------- Failed to calculate pseudo closed-loop weights or comfort:")
        traceback.print_exc()
        pdm_score_df["weight"] = 1.0
        pseudo_closed_loop_valid = False

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    pcl_group_score, pcl_stage1_score, pcl_stage2_score = calculate_individual_mapping_scores(
        pdm_score_df[score_cols + ["token", "weight"]], all_mappings
    )

    for col in score_cols:
        stage_one_mask = pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
        stage_two_mask = pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC

        pdm_score_df.loc[stage_one_mask, f"{col}_stage_one"] = pdm_score_df.loc[stage_one_mask, col]
        pdm_score_df.loc[stage_two_mask, f"{col}_stage_two"] = pdm_score_df.loc[stage_two_mask, col]

    pdm_score_df.drop(columns=score_cols, inplace=True)
    pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(pdm_score_df["score_stage_two"])
    pdm_score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)

    stage1_cols = [f"{col}_stage_one" for col in score_cols if col != "score"]
    stage2_cols = [f"{col}_stage_two" for col in score_cols if col != "score"]
    score_cols = stage1_cols + stage2_cols + ["score"]

    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]

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
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final extended pdm score of valid results: {pdm_score_df[pdm_score_df["token"] == "extended_pdm_score_combined"]["score"].iloc[0]}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )

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
