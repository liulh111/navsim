from __future__ import annotations

import importlib.util
import json
import logging
import os
import pickle
import subprocess
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import hydra
import pandas as pd
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.script.builders.worker_pool_builder import build_worker

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/gpudrive_json_export"
CONFIG_NAME = "default_run_gpudrive_json_export"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_workspace_path(path: Any) -> Path:
    resolved = Path(str(path)).expanduser()
    if not resolved.is_absolute():
        resolved = _workspace_root() / resolved
    return resolved.resolve()


def _load_single_token_exporter() -> Any:
    """Load the NAVSIM synthetic-scene exporter."""
    repo_root = Path(__file__).resolve().parents[3]
    exporter_path = repo_root / "scripts" / "export_navsim_scene_gpudrive_json_logs.py"
    if not exporter_path.is_file():
        raise FileNotFoundError(f"Could not find single-token exporter at {exporter_path}")

    spec = importlib.util.spec_from_file_location("navsim_single_token_gpudrive_json_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import exporter module from {exporter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_output_dir(cfg: DictConfig, stage: str) -> Path:
    output_dir = Path(cfg.gpudrive_json_output_dir)
    if bool(cfg.separate_stage_dirs):
        output_dir = output_dir / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _output_path_for_token(cfg: DictConfig, scenario_json: Dict[str, Any], stage: str) -> Path:
    return _stage_output_dir(cfg, stage) / scenario_json["name"]


def _load_reference_json_for_token(cfg: DictConfig, exporter: Any, token: str) -> tuple[dict[str, Any] | None, str]:
    reference_json_dir = cfg.get("reference_json_dir")
    if not reference_json_dir:
        return None, ""

    if str(reference_json_dir) == "auto":
        reference_dir = (
            Path(exporter._workspace_root()) / "ScenarioMax" / "tmp" / "token_obs_export"  # noqa: SLF001
        )
    else:
        reference_dir = Path(str(reference_json_dir)).expanduser()

    reference_name = str(cfg.reference_json_pattern).format(
        dataset_name=str(cfg.dataset_name),
        dataset_version=str(cfg.dataset_version),
        token=token,
    )
    reference_path = reference_dir / reference_name
    if not reference_path.is_file():
        return None, ""

    return exporter._load_reference_json(reference_path), str(reference_path)  # noqa: SLF001


def _resolve_goal_source_for_stage(cfg: DictConfig, stage: str) -> str:
    goal_source = str(cfg.goal_source)
    if goal_source != "auto":
        return goal_source
    return "future" if stage == "stage_one" else "route_local"


def _split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
    return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]


def _iter_stage_one_records(cfg: DictConfig, data_point: Dict[str, Any]) -> List[tuple[str, Dict[str, Any]]]:
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    requested_tokens = set(data_point["tokens"])
    log_name = data_point["log_file"]
    log_path = Path(cfg.navsim_log_path) / f"{log_name}.pkl"

    with log_path.open("rb") as file:
        scene_dict_list = pickle.load(file)
    records: List[tuple[str, Dict[str, Any]]] = []
    for frame_list in _split_list(scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval):
        if len(frame_list) < scene_filter.num_frames:
            continue
        if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
            continue

        token = frame_list[scene_filter.num_history_frames - 1]["token"]
        if token not in requested_tokens:
            continue

        current_frame = frame_list[scene_filter.num_history_frames - 1]
        records.append(
            (
                token,
                {
                    "log_name": current_frame["log_name"],
                    "timestamp": int(current_frame["timestamp"]),
                    "map_name": current_frame["map_location"],
                },
            )
        )
    return records


def _iter_stage_two_scenes(cfg: DictConfig, data_point: Dict[str, Any]) -> List[tuple[str, Scene]]:
    scenes: List[tuple[str, Scene]] = []
    for token, scene_path in data_point["items"]:
        scenes.append(
            (
                token,
                Scene.load_from_disk(
                    file_path=Path(scene_path),
                    sensor_blobs_path=Path(cfg.synthetic_sensor_path),
                    sensor_config=SensorConfig.build_no_sensors(),
                ),
            )
        )
    return scenes


def _export_scene(
    cfg: DictConfig,
    exporter: Any,
    token: str,
    stage: str,
    scene: Scene,
) -> pd.DataFrame:
    row: Dict[str, Any] = {
        "token": token,
        "stage": stage,
        "valid": False,
        "json_path": "",
        "num_objects": 0,
        "num_roads": 0,
        "num_tl_states": 0,
        "sdc_track_index": -1,
        "reference_json_path": "",
        "used_reference_order": False,
        "goal_source": "",
        "error": "",
    }

    try:
        goal_source = _resolve_goal_source_for_stage(cfg, stage)
        row["goal_source"] = goal_source
        reference_json, reference_json_path = (
            _load_reference_json_for_token(cfg, exporter, token) if bool(cfg.align_reference_order) else (None, "")
        )
        scenario_json = exporter.build_scenario_json(
            scene,
            map_radius=int(cfg.map_radius),
            goal_source=goal_source,
            dataset_name=str(cfg.dataset_name),
            dataset_version=str(cfg.dataset_version),
            scenario_type_prefix=str(cfg.scenario_type_prefix),
            reference_json=reference_json,
            align_reference_order=bool(cfg.align_reference_order),
            target_interval=float(cfg.target_interval),
        )
        output_path = _output_path_for_token(cfg, scenario_json, stage)
        row["json_path"] = str(output_path)
        row["reference_json_path"] = reference_json_path
        row["used_reference_order"] = reference_json is not None

        if output_path.exists() and not bool(cfg.overwrite):
            row["valid"] = True
            row["skipped_existing"] = True
        else:
            with output_path.open("w", encoding="utf-8") as file:
                if bool(cfg.pretty_json):
                    json.dump(exporter._jsonable(scenario_json), file, indent=2)
                else:
                    json.dump(exporter._jsonable(scenario_json), file)
            row["valid"] = True
            row["skipped_existing"] = False

        row["num_objects"] = len(scenario_json["objects"])
        row["num_roads"] = len(scenario_json["roads"])
        row["num_tl_states"] = len(scenario_json["tl_states"])
        row["sdc_track_index"] = scenario_json["metadata"]["sdc_track_index"]
        row["num_steps"] = (
            len(scenario_json["objects"][0]["position"]) if scenario_json["objects"] else 0
        )
        row["source"] = scenario_json["metadata"].get("source", "synthetic_scene")

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to export token %s", token)
        traceback.print_exc()
        row["error"] = repr(exc)

    return pd.DataFrame([row])


def _export_stage_one_record(
    cfg: DictConfig,
    token: str,
    record: Dict[str, Any],
) -> pd.DataFrame:
    row: Dict[str, Any] = {
        "token": token,
        "stage": "stage_one",
        "valid": False,
        "json_path": "",
        "num_objects": 0,
        "num_roads": 0,
        "num_tl_states": 0,
        "sdc_track_index": -1,
        "reference_json_path": "",
        "used_reference_order": False,
        "goal_source": "scenario_max",
        "num_steps": 0,
        "source": "nuplan_db",
        "scenariomax_log_path": "",
        "error": "",
    }

    try:
        output_dir = _stage_output_dir(cfg, "stage_one")
        output_path = output_dir / f"{cfg.dataset_name}_{cfg.dataset_version}_{token}.json"
        log_path = output_path.with_suffix(".log")
        row["json_path"] = str(output_path)
        row["scenariomax_log_path"] = str(log_path)

        if output_path.exists() and not bool(cfg.overwrite):
            with output_path.open("r", encoding="utf-8") as file:
                scenario_json = json.load(file)
            row["skipped_existing"] = True
        else:
            row["skipped_existing"] = False
            scenariomax_root = _resolve_workspace_path(cfg.scenariomax_root)
            exporter_path = scenariomax_root / "scripts" / "export_nuplan_token_obs.py"
            scenariomax_python = _resolve_workspace_path(cfg.scenariomax_python)
            if not exporter_path.is_file():
                raise FileNotFoundError(f"Could not find ScenarioMax exporter at {exporter_path}")
            if not scenariomax_python.is_file():
                raise FileNotFoundError(
                    f"Could not find ScenarioMax Python at {scenariomax_python}. "
                    "Set SCENARIOMAX_PYTHON or override scenariomax_python."
                )

            with tempfile.TemporaryDirectory(prefix=f"scenariomax_{token}_", dir="/tmp") as temp_dir:
                command = [
                    str(scenariomax_python),
                    str(exporter_path),
                    "--nuplan-data-root",
                    str(_resolve_workspace_path(cfg.nuplan_data_root)),
                    "--nuplan-maps-root",
                    str(_resolve_workspace_path(cfg.nuplan_maps_root)),
                    "--log-name",
                    str(record["log_name"]),
                    "--token",
                    token,
                    "--timestamp",
                    str(record["timestamp"]),
                    "--map-name",
                    str(record["map_name"]),
                    "--map-version",
                    "nuplan-maps-v1.0",
                    "--scenario-duration",
                    str(cfg.scenario_duration),
                    "--subsample-ratio",
                    str(cfg.subsample_ratio),
                    "--dataset-version",
                    str(cfg.dataset_version),
                    "--output-dir",
                    temp_dir,
                    "--export-mode",
                    "json",
                ]
                process = subprocess.run(
                    command,
                    cwd=scenariomax_root,
                    env={**os.environ, "MPLCONFIGDIR": "/tmp"},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                log_path.write_text(
                    f"Command: {' '.join(command)}\n\n"
                    f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}",
                    encoding="utf-8",
                )
                if process.returncode != 0:
                    raise RuntimeError(
                        f"ScenarioMax exited with code {process.returncode}; see {log_path}"
                    )

                generated_path = Path(temp_dir) / output_path.name
                if not generated_path.is_file():
                    matches = list(Path(temp_dir).glob(f"*_{cfg.dataset_version}_{token}.json"))
                    if len(matches) != 1:
                        raise FileNotFoundError(
                            f"ScenarioMax did not produce a JSON for {token}; see {log_path}"
                        )
                    generated_path = matches[0]
                with generated_path.open("r", encoding="utf-8") as file:
                    scenario_json = json.load(file)

        num_steps = len(scenario_json["objects"][0]["position"]) if scenario_json["objects"] else 0
        expected_steps = int(
            round(float(cfg.scenario_duration) / (0.05 / float(cfg.subsample_ratio)))
        )
        if num_steps != expected_steps or num_steps > 91:
            raise ValueError(
                f"ScenarioMax Stage One export must contain {expected_steps} steps "
                f"and at most 91, got {num_steps}"
            )

        if not row["skipped_existing"]:
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(
                    scenario_json,
                    file,
                    indent=2 if bool(cfg.pretty_json) else None,
                )

        row["valid"] = True
        row["num_objects"] = len(scenario_json["objects"])
        row["num_roads"] = len(scenario_json["roads"])
        row["num_tl_states"] = len(scenario_json["tl_states"])
        row["sdc_track_index"] = scenario_json["metadata"]["sdc_track_index"]
        row["num_steps"] = num_steps
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to export Stage One token %s", token)
        traceback.print_exc()
        row["valid"] = False
        row["error"] = repr(exc)

    return pd.DataFrame([row])


def export_gpudrive_json(args: List[Dict[str, Any]]) -> List[pd.DataFrame]:
    """
    Worker entrypoint for exporting NAVHARD scenes to GPUDrive JSON.
    :param args: grouped log/tokens/cfg payloads from worker_map.
    :return: one-row DataFrames for the export manifest.
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting GPUDrive JSON export worker thread_id={thread_id}, node_id={node_id}")

    cfg: DictConfig = args[0]["cfg"]  # type: ignore[assignment]
    exporter = _load_single_token_exporter()
    rows: List[pd.DataFrame] = []

    for data_point in args:
        try:
            kind = data_point["kind"]
            stage = "stage_one" if kind == "stage_one" else "stage_two"
            if kind == "stage_one":
                records = _iter_stage_one_records(cfg, data_point)
                for idx, (token, record) in enumerate(records):
                    logger.info(
                        "Exporting %s %s (%s/%s) in thread_id=%s, node_id=%s",
                        stage,
                        token,
                        idx + 1,
                        len(records),
                        thread_id,
                        node_id,
                    )
                    rows.append(
                        _export_stage_one_record(
                            cfg,
                            token,
                            record,
                        )
                    )
                continue

            scenes = _iter_stage_two_scenes(cfg, data_point)
            for idx, (token, scene) in enumerate(scenes):
                logger.info(
                    "Exporting %s %s (%s/%s) in thread_id=%s, node_id=%s",
                    stage,
                    token,
                    idx + 1,
                    len(scenes),
                    thread_id,
                    node_id,
                )
                rows.append(_export_scene(cfg, exporter, token, stage, scene))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process export data point: %s", data_point)
            traceback.print_exc()
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "token": "",
                            "stage": data_point.get("kind", ""),
                            "valid": False,
                            "json_path": "",
                            "num_objects": 0,
                            "num_roads": 0,
                            "num_tl_states": 0,
                            "sdc_track_index": -1,
                            "reference_json_path": "",
                            "used_reference_order": False,
                            "goal_source": "",
                            "num_steps": 0,
                            "source": "",
                            "scenariomax_log_path": "",
                            "error": repr(exc),
                        }
                    ]
                )
            )

    return rows


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entrypoint for batch NAVSIM -> GPUDrive JSON export."""
    build_logger(cfg)
    worker = build_worker(cfg)

    logger.info("Building scene index for GPUDrive JSON export...")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if str(cfg.export_stage) == "stage_one":
        scene_filter.include_synthetic_scenes = False

    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    if str(cfg.export_stage) == "stage_one":
        allowed_tokens = set(scene_loader.tokens_stage_one)
    elif str(cfg.export_stage) == "stage_two":
        allowed_tokens = set(scene_loader.synthetic_scenes_tokens)
    elif str(cfg.export_stage) == "reactive_stage_two":
        allowed_tokens = set(scene_loader.reactive_tokens_stage_two or [])
    elif str(cfg.export_stage) == "non_reactive_stage_two":
        allowed_tokens = set(scene_loader.non_reactive_tokens_stage_two or [])
    else:
        allowed_tokens = set(scene_loader.tokens)

    data_points = []
    if str(cfg.export_stage) in {"all", "stage_one"}:
        stage_one_tokens_by_log: Dict[str, List[str]] = {}
        for token, frame_list in scene_loader.scene_frames_dicts.items():
            if token not in allowed_tokens:
                continue
            log_name = frame_list[0]["log_name"]
            stage_one_tokens_by_log.setdefault(log_name, []).append(token)
        data_points.extend(
            {
                "cfg": cfg,
                "kind": "stage_one",
                "log_file": log_file,
                "tokens": sorted(tokens),
            }
            for log_file, tokens in stage_one_tokens_by_log.items()
        )

    if str(cfg.export_stage) in {"all", "stage_two", "reactive_stage_two", "non_reactive_stage_two"}:
        stage_two_items_by_log: Dict[str, List[tuple[str, str]]] = {}
        for token, (scene_path, log_name) in scene_loader.synthetic_scenes.items():
            if token not in allowed_tokens:
                continue
            stage_two_items_by_log.setdefault(log_name, []).append((token, str(scene_path)))
        data_points.extend(
            {
                "cfg": cfg,
                "kind": "stage_two",
                "log_file": log_file,
                "items": sorted(items),
            }
            for log_file, items in stage_two_items_by_log.items()
        )

    num_scenes = sum(
        len(data_point["tokens"]) if data_point["kind"] == "stage_one" else len(data_point["items"])
        for data_point in data_points
    )
    logger.info(
        "Starting GPUDrive JSON export for %d scenes across %d groups into %s",
        num_scenes,
        len(data_points),
        cfg.gpudrive_json_output_dir,
    )

    rows: List[pd.DataFrame] = worker_map(worker, export_gpudrive_json, data_points)
    manifest = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    manifest_path = Path(cfg.gpudrive_json_output_dir) / "export_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    failed = manifest[~manifest["valid"]] if not manifest.empty else manifest
    if not failed.empty:
        failed_path = Path(cfg.gpudrive_json_output_dir) / "failed_exports.csv"
        failed.to_csv(failed_path, index=False)
        logger.warning("GPUDrive JSON export completed with %d failed scenes: %s", len(failed), failed_path)
    else:
        logger.info("GPUDrive JSON export completed successfully.")

    logger.info("Export manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
