#!/usr/bin/env python
"""Roll out a continuous-action policy on NAVHARD Stage One GPUDrive JSON."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GPUDRIVE_ROOT = WORKSPACE_ROOT / "gpudrive"
if str(GPUDRIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(GPUDRIVE_ROOT))

import numpy as np
import torch

TOKEN_RE = re.compile(r"([0-9a-fA-F]{16,})$")
ROLLOUT_STEPS = 40
NAVSIM_SAMPLE_INDICES = np.arange(5, ROLLOUT_STEPS + 1, 5)


@dataclass
class ListSceneDataLoader:
    """Single-batch loader accepted by GPUDriveTorchEnv."""

    scenes: Sequence[str]

    def __post_init__(self) -> None:
        self.batch_size = len(self.scenes)
        self.dataset_size = len(self.scenes)
        self._used = False

    def __iter__(self) -> Iterator[list[str]]:
        self._used = False
        return self

    def __next__(self) -> list[str]:
        if self._used:
            raise StopIteration
        self._used = True
        return list(self.scenes)

    def __len__(self) -> int:
        return 1


class ContinuousPolicyAgent(Protocol):
    """Policy interface for acceleration and steering actions."""

    name: str

    def act(self, obs: torch.Tensor, step_index: int) -> torch.Tensor:
        """Return [num_controlled_agents, 2] acceleration/steering actions."""


class ExpertPolicyAgent:
    """GPUDrive inverse-dynamics expert behind the common policy interface."""

    name = "expert_bicycle"

    def __init__(self, expert_actions: torch.Tensor, controlled_mask: torch.Tensor):
        self._expert_actions = expert_actions
        self._controlled_mask = controlled_mask

    def act(self, obs: torch.Tensor, step_index: int) -> torch.Tensor:
        actions = self._expert_actions[:, :, step_index, :2][self._controlled_mask]
        if actions.shape != (obs.shape[0], 2):
            raise ValueError(f"Policy action shape {tuple(actions.shape)} does not match obs batch {obs.shape[0]}")
        return actions


def _infer_bicycle_actions(log_velocities: torch.Tensor, log_yaws: torch.Tensor) -> torch.Tensor:
    """Invert GPUDrive's bicycle equations using recorded heading rather than velocity direction."""
    dt = 0.1
    speeds = torch.linalg.vector_norm(log_velocities, dim=-1)
    yaw = log_yaws[..., 0]
    acceleration = torch.clamp((speeds[..., 1:] - speeds[..., :-1]) / dt, -6.0, 6.0)
    distance = speeds[..., :-1] * dt + 0.5 * acceleration * dt * dt
    delta_yaw = _wrap_angle(yaw[..., 1:] - yaw[..., :-1])
    steering = torch.where(
        torch.abs(distance) > 1e-6,
        delta_yaw / distance,
        torch.zeros_like(delta_yaw),
    )
    steering = torch.clamp(steering, -3.0, 3.0)
    return torch.stack((acceleration, steering), dim=-1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results/gpudrive_json/navhard_two_stage/stage_one",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results/gpudrive_rollouts/navhard_stage_one/expert_bicycle",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-position-error", type=float, default=5.0)
    parser.add_argument("--max-heading-error", type=float, default=0.1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-jsons", nargs="*", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _token_from_json(path: Path) -> str:
    match = TOKEN_RE.search(path.stem)
    return match.group(1).lower() if match else path.stem


def _json_files(json_dir: Path, max_scenes: int | None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(json_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict) and "objects" in data and "roads" in data:
            files.append(path)
    if max_scenes is not None:
        files = files[:max_scenes]
    if not files:
        raise FileNotFoundError(f"No GPUDrive scene JSON files found in {json_dir}")
    return files


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _localize_poses(global_poses: torch.Tensor) -> np.ndarray:
    origin = global_poses[:, 0]
    delta_xy = global_poses[:, :, :2] - origin[:, None, :2]
    cos_yaw = torch.cos(origin[:, 2])
    sin_yaw = torch.sin(origin[:, 2])
    local_x = cos_yaw[:, None] * delta_xy[:, :, 0] + sin_yaw[:, None] * delta_xy[:, :, 1]
    local_y = -sin_yaw[:, None] * delta_xy[:, :, 0] + cos_yaw[:, None] * delta_xy[:, :, 1]
    local_yaw = _wrap_angle(global_poses[:, :, 2] - origin[:, None, 2])
    return torch.stack((local_x, local_y, local_yaw), dim=-1).cpu().numpy().astype(np.float32)


def _sdc_global_poses(env) -> torch.Tensor:
    from gpudrive.datatypes.observation import GlobalEgoState

    state = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(),
        backend="torch",
        device=env.device,
    )
    return torch.stack((state.pos_x[:, 0], state.pos_y[:, 0], state.rotation_angle[:, 0]), dim=-1)


def _make_env(paths: Sequence[Path], device: str):
    from gpudrive.env.config import EnvConfig
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    config = EnvConfig(
        num_worlds=len(paths),
        dynamics_model="bicycle",
        collision_behavior="ignore",
        init_mode="all_valid",
        init_steps=0,
        dist_to_goal_threshold=-1.0,
    )
    return GPUDriveTorchEnv(
        config=config,
        data_loader=ListSceneDataLoader([str(path) for path in paths]),
        max_cont_agents=1,
        device=device,
        action_type="discrete",
    )


def _rollout_batch(paths: Sequence[Path], args: argparse.Namespace) -> list[dict[str, object]]:
    from gpudrive.datatypes.metadata import Metadata

    env = _make_env(paths, args.device)
    try:
        obs = env.reset(env.cont_agent_mask)
        controlled_mask = env.cont_agent_mask
        metadata = Metadata.from_tensor(env.sim.metadata_tensor()).is_sdc.to(controlled_mask.device)
        controlled_per_world = controlled_mask.sum(dim=1)
        if not torch.all(controlled_per_world == 1):
            raise ValueError(f"Expected one controlled SDC per world, got {controlled_per_world.tolist()}")
        if not torch.all(controlled_mask[:, 0]) or not torch.all(metadata[:, 0] == 1):
            raise ValueError("GPUDrive agent 0 is not the controlled SDC in every world")

        _, expert_positions, expert_velocities, expert_yaws = env.get_expert_actions()
        expert_actions = _infer_bicycle_actions(expert_velocities, expert_yaws)
        policy: ContinuousPolicyAgent = ExpertPolicyAgent(expert_actions, controlled_mask)
        frames = [_sdc_global_poses(env)]

        for step_index in range(ROLLOUT_STEPS):
            policy_actions = policy.act(obs, step_index)
            action_tensor = torch.zeros(
                (env.num_worlds, env.max_agent_count, 3),
                dtype=policy_actions.dtype,
                device=env.device,
            )
            action_tensor[controlled_mask, :2] = policy_actions
            env.step_dynamics(action_tensor)
            frames.append(_sdc_global_poses(env))

            dones = env.get_dones().bool()
            if torch.any(dones & controlled_mask):
                failed_worlds = torch.where((dones & controlled_mask).any(dim=1))[0].tolist()
                raise RuntimeError(f"Controlled SDC terminated before 4.0s in worlds {failed_worlds}")
            obs = env.get_obs(controlled_mask)

        global_poses = torch.stack(frames, dim=1)
        log_positions = expert_positions[:, 0, : ROLLOUT_STEPS + 1]
        log_yaws = expert_yaws[:, 0, : ROLLOUT_STEPS + 1, 0]
        position_errors = torch.linalg.vector_norm(global_poses[:, :, :2] - log_positions, dim=-1)
        heading_errors = torch.abs(_wrap_angle(global_poses[:, :, 2] - log_yaws))
        max_position_errors = position_errors.max(dim=1).values
        max_heading_errors = heading_errors.max(dim=1).values
        if torch.any(max_position_errors > args.max_position_error):
            print(f"Expert position reconstruction error exceeded limit: {max_position_errors.tolist()}")
        if torch.any(max_heading_errors > args.max_heading_error):
            print(f"Expert heading reconstruction error exceeded limit: {max_heading_errors.tolist()}")

        local_poses = _localize_poses(global_poses)
        if not np.allclose(local_poses[:, 0], 0.0, atol=1e-5):
            raise ValueError("Localized rollout does not start at [0, 0, 0]")

        rows: list[dict[str, object]] = []
        raw_dir = args.output_dir / "raw"
        trajectory_dir = args.output_dir / "trajectories"
        for world_index, path in enumerate(paths):
            token = _token_from_json(path)
            raw_path = raw_dir / f"{token}.npy"
            trajectory_path = trajectory_dir / f"{token}.npy"
            np.save(raw_path, local_poses[world_index])
            np.save(trajectory_path, local_poses[world_index, NAVSIM_SAMPLE_INDICES])
            rows.append(
                {
                    "token": token,
                    "policy": policy.name,
                    "json_path": str(path),
                    "raw_path": str(raw_path),
                    "trajectory_path": str(trajectory_path),
                    "num_raw_states": ROLLOUT_STEPS + 1,
                    "num_navsim_poses": len(NAVSIM_SAMPLE_INDICES),
                    "max_position_error": float(max_position_errors[world_index].item()),
                    "max_heading_error": float(max_heading_errors[world_index].item()),
                }
            )
        return rows
    finally:
        env.close()


def _run_worker(paths: Sequence[Path], args: argparse.Namespace) -> list[dict[str, object]]:
    with tempfile.NamedTemporaryFile(prefix="navhard_rollout_", suffix=".json", delete=False) as file:
        result_path = Path(file.name)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-result",
        str(result_path),
        "--json-dir",
        str(args.json_dir),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--max-position-error",
        str(args.max_position_error),
        "--max-heading-error",
        str(args.max_heading_error),
        "--worker-jsons",
        *[str(path) for path in paths],
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        result_path.unlink(missing_ok=True)
        raise RuntimeError(f"GPUDrive rollout worker failed with exit code {result.returncode}")
    with result_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    result_path.unlink(missing_ok=True)
    return rows


def _write_manifest(rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "token",
        "policy",
        "json_path",
        "raw_path",
        "trajectory_path",
        "num_raw_states",
        "num_navsim_poses",
        "max_position_error",
        "max_heading_error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_position_error < 0 or args.max_heading_error < 0:
        raise ValueError("Error thresholds must be non-negative")

    (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    if args.worker:
        if not args.worker_jsons or args.worker_result is None:
            raise ValueError("Worker mode requires --worker-jsons and --worker-result")
        rows = _rollout_batch([Path(path) for path in args.worker_jsons], args)
        with args.worker_result.open("w", encoding="utf-8") as file:
            json.dump(rows, file)
        return

    files = _json_files(args.json_dir, args.max_scenes)
    rows: list[dict[str, object]] = []
    pending: list[Path] = []
    for path in files:
        token = _token_from_json(path)
        raw_path = args.output_dir / "raw" / f"{token}.npy"
        trajectory_path = args.output_dir / "trajectories" / f"{token}.npy"
        if raw_path.is_file() and trajectory_path.is_file() and not args.overwrite:
            raise FileExistsError(f"Rollout already exists for {token}; use --overwrite")
        pending.append(path)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        rows.extend(_run_worker(batch, args))
        _write_manifest(rows, args.output_dir)
        print(f"[{len(rows)}/{len(pending)}] completed", flush=True)

    if len(rows) != len(files):
        raise RuntimeError(f"Exported {len(rows)} trajectories for {len(files)} JSON files")
    print(f"Wrote {len(rows)} rollouts to {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
