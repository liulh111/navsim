# import debugpy
# debugpy.listen(("localhost", 9501))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

import os
import logging
import numpy as np
import hydra
from hydra.utils import instantiate

from dataclasses import replace
from pathlib import Path
from omegaconf import DictConfig
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navsim.visualization.plots import (
    configure_bev_ax,
    plot_cameras_frame,
    plot_cameras_frame_with_annotations,
    frame_plot_to_gif,
)
from navsim.visualization.bev import add_annotations_to_bev_ax

from navsim.planning.script.extract_waymo_feature_stage_one import extract_waymo_whitebox_features  # noqa: F401
from navsim.planning.script.visualization import visualize_single_obs_and_save  # noqa: F401


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _build_synthetic_sensor_config() -> SensorConfig:
    """Synthetic stage-two sensor_blobs only contain cameras (no lidar)."""
    cfg = SensorConfig.build_all_sensors(include=True)
    return replace(cfg, lidar_pc=False)


def _coerce_annotation_names_to_array(scene: Scene) -> None:
    """Synthetic scenes store annotations.names as a Python list; downstream
    camera overlay code indexes it with a boolean numpy array, which requires
    an ndarray. Coerce in-place for every frame that has annotations."""
    for frame in scene.frames:
        anns = getattr(frame, "annotations", None)
        if anns is None:
            continue
        if not isinstance(anns.names, np.ndarray):
            anns.names = np.array(anns.names, dtype=object)


def navsim_stage_two_visualization(scene: Scene, token: str) -> None:
    _coerce_annotation_names_to_array(scene)

    VIZ_DIR = Path(os.path.join("viz_output", "stage_two", token))
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    current_frame_idx = scene.scene_metadata.num_history_frames - 1

    # --- 1. 仅标注框的自定义 BEV ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_title(f"BEV Annotations  |  token: {token[:12]}...")
    add_annotations_to_bev_ax(ax, scene.frames[current_frame_idx].annotations)
    configure_bev_ax(ax)
    out_path = VIZ_DIR / f"bev_annotations.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved BEV annotations -> {out_path}")

    # --- 2. 相机 3×3 网格图（中心为 BEV）---
    fig, ax = plot_cameras_frame(scene, current_frame_idx)
    out_path = VIZ_DIR / f"cameras.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cameras frame -> {out_path}")

    # --- 3. 相机图 + bounding-box 标注 ---
    fig, ax = plot_cameras_frame_with_annotations(scene, current_frame_idx)
    out_path = VIZ_DIR / f"cameras_annotations.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cameras+annotations -> {out_path}")

    # NOTE: lidar-dependent plots (plot_cameras_frame_with_lidar,
    # add_lidar_to_bev_ax) are skipped — synthetic sensor_blobs have no lidar.

    # --- 4. 所有帧的动态 GIF（以相机+标注为例）---
    frame_indices = list(range(len(scene.frames)))
    gif_path = str(VIZ_DIR / f"cameras_annotations.gif")
    frame_plot_to_gif(
        gif_path,
        plot_cameras_frame_with_annotations,
        scene,
        frame_indices,
    )
    logger.info(f"Saved GIF -> {gif_path}")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for stage-two (synthetic) whitebox visualization.
    :param cfg: omegaconf dictionary
    """

    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=_build_synthetic_sensor_config(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_loader_tokens_stage_two = scene_loader.reactive_tokens_stage_two
    tokens_to_evaluate_stage_two = sorted(set(scene_loader_tokens_stage_two) & set(metric_cache_loader.tokens))

    token = tokens_to_evaluate_stage_two[0]
    scene = scene_loader.get_scene_from_token(token)

    navsim_stage_two_visualization(scene, token)

    # VIZ_DIR = Path(os.path.join("viz_output_stage_two", token))
    # save_path = VIZ_DIR / f"waymo_whitebox_obs.png"
    # ego_state, partner_state, road_state = extract_waymo_whitebox_features(scene)
    # visualize_single_obs_and_save(
    #     ego=ego_state, partners=partner_state, roads=road_state, save_path=str(save_path)
    # )


if __name__ == "__main__":
    main()
