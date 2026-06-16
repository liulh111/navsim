#!/usr/bin/env python
"""Roll out NAVHARD Stage One/Two GPUDrive JSON with ego plus NAVSIM-IDM actors."""

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
from typing import Any, Iterator, Protocol, Sequence

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


def _agent_id(raw_id: int | float) -> int:
    return int(round(float(raw_id)))


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

    def act(self, env: Any, obs: torch.Tensor, step_index: int) -> torch.Tensor:
        """Return [num_worlds, 2] acceleration/steering actions for ego."""


class ExpertPolicyAgent:
    """GPUDrive inverse-dynamics expert behind the common policy interface."""

    name = "expert_bicycle"

    def __init__(self, expert_actions: torch.Tensor, controlled_mask: torch.Tensor):
        self._expert_actions = expert_actions
        self._controlled_mask = controlled_mask

    def act(self, env: Any, obs: torch.Tensor, step_index: int) -> torch.Tensor:
        actions = self._expert_actions[:, 0, step_index, :2]
        if actions.shape != (env.num_worlds, 2):
            raise ValueError(f"Policy action shape {tuple(actions.shape)} does not match worlds {env.num_worlds}")
        return actions


class ZeroPolicyAgent:
    """Stationary ego actor for smoke tests."""

    name = "zero"

    def act(self, env: Any, obs: torch.Tensor, step_index: int) -> torch.Tensor:
        return torch.zeros((env.num_worlds, 2), dtype=torch.float32, device=env.device)


@dataclass(frozen=True)
class IDMVehicleRoute:
    agent_id: int
    route_xy: torch.Tensor
    route_heading: torch.Tensor
    route_progress: torch.Tensor
    progress: float
    velocity: float


@dataclass
class WorldActorMapping:
    ego_index: int
    idm_indices: list[int]
    static_vehicle_indices: list[int]
    idm_agent_ids: set[int]


class IDMActor:
    """Python/Torch NAVSIM-style IDM actor for sidecar vehicles."""

    def __init__(
        self,
        min_gap: float = 1.0,
        headway_time: float = 1.5,
        accel_max: float = 1.0,
        decel_max: float = 2.0,
        target_velocity: float = 10.0,
        dt: float = 0.1,
        leader_lateral_slack: float = 1.0,
    ) -> None:
        self.min_gap = min_gap
        self.headway_time = headway_time
        self.accel_max = accel_max
        self.decel_max = decel_max
        self.target_velocity = target_velocity
        self.dt = dt
        self.leader_lateral_slack = leader_lateral_slack
        self._routes: list[dict[int, IDMVehicleRoute]] = []
        self._mappings: list[WorldActorMapping] = []

    def reset(self, sidecars: Sequence[dict[str, Any]], mappings: Sequence[WorldActorMapping], device: torch.device) -> None:
        self._mappings = list(mappings)
        self._routes = []
        for sidecar in sidecars:
            world_routes: dict[int, IDMVehicleRoute] = {}
            for vehicle in sidecar.get("vehicles", {}).values():
                route_xy = torch.as_tensor(vehicle["route_xy"], dtype=torch.float32, device=device)
                route_heading = torch.as_tensor(vehicle["route_heading"], dtype=torch.float32, device=device)
                route_progress = torch.as_tensor(vehicle["route_progress"], dtype=torch.float32, device=device)
                if route_xy.ndim != 2 or route_xy.shape[0] < 2:
                    continue
                agent_id = _agent_id(vehicle["agent_id"])
                world_routes[agent_id] = IDMVehicleRoute(
                    agent_id=agent_id,
                    route_xy=route_xy,
                    route_heading=route_heading,
                    route_progress=route_progress,
                    progress=float(vehicle.get("initial_progress", route_progress[0].item())),
                    velocity=0.0,
                )
            self._routes.append(world_routes)

    def act(self, env: Any) -> dict[tuple[int, int], tuple[float, float]]:
        from gpudrive.datatypes.observation import GlobalEgoState, LocalEgoState

        global_state = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(),
            backend="torch",
            device=env.device,
        )
        local_state = LocalEgoState.from_tensor(
            env.sim.self_observation_tensor(),
            backend="torch",
            device=env.device,
        )

        actions: dict[tuple[int, int], tuple[float, float]] = {}
        for world_idx, mapping in enumerate(self._mappings):
            vehicle_indices = [mapping.ego_index, *mapping.idm_indices, *mapping.static_vehicle_indices]
            for agent_idx in mapping.idm_indices:
                agent_id = _agent_id(global_state.id[world_idx, agent_idx].item())
                route = self._routes[world_idx].get(agent_id)
                if route is None:
                    actions[(world_idx, agent_idx)] = (0.0, 0.0)
                    continue

                current_speed = float(local_state.speed[world_idx, agent_idx].item())
                route = IDMVehicleRoute(
                    agent_id=route.agent_id,
                    route_xy=route.route_xy,
                    route_heading=route.route_heading,
                    route_progress=route.route_progress,
                    progress=max(route.progress, self._project_progress(route, global_state, world_idx, agent_idx)[0]),
                    velocity=max(0.0, current_speed),
                )
                leader_gap, leader_speed = self._nearest_leader(route, global_state, local_state, world_idx, vehicle_indices, agent_idx)
                accel = self._idm_acceleration(route.velocity, leader_gap, leader_speed)
                next_velocity = max(0.0, route.velocity + accel * self.dt)
                distance = route.velocity * self.dt + 0.5 * accel * self.dt * self.dt
                next_progress = max(route.progress, route.progress + max(0.0, distance))
                target_heading = self._interpolate_heading(route, next_progress)
                current_heading = float(global_state.rotation_angle[world_idx, agent_idx].item())
                steering = 0.0 if abs(distance) < 1e-6 else float(_wrap_angle(torch.tensor(target_heading - current_heading)).item()) / distance
                accel = float(np.clip(accel, -6.0, 6.0))
                steering = float(np.clip(steering, -3.0, 3.0))

                self._routes[world_idx][agent_id] = IDMVehicleRoute(
                    agent_id=route.agent_id,
                    route_xy=route.route_xy,
                    route_heading=route.route_heading,
                    route_progress=route.route_progress,
                    progress=next_progress,
                    velocity=next_velocity,
                )
                actions[(world_idx, agent_idx)] = (accel, steering)
        return actions

    def _idm_acceleration(self, velocity: float, leader_gap: float | None, leader_speed: float | None) -> float:
        free_road_term = (velocity / max(self.target_velocity, 1e-3)) ** 4
        interaction_term = 0.0
        if leader_gap is not None and leader_speed is not None:
            closing_speed = velocity - leader_speed
            desired_gap = self.min_gap + max(
                0.0,
                velocity * self.headway_time
                + velocity * closing_speed / (2.0 * np.sqrt(max(self.accel_max * self.decel_max, 1e-6))),
            )
            interaction_term = (desired_gap / max(leader_gap, 1e-3)) ** 2
        return self.accel_max * (1.0 - free_road_term - interaction_term)

    def _nearest_leader(
        self,
        route: IDMVehicleRoute,
        global_state: Any,
        local_state: Any,
        world_idx: int,
        vehicle_indices: Sequence[int],
        follower_idx: int,
    ) -> tuple[float | None, float | None]:
        follower_progress = route.progress
        best_gap: float | None = None
        best_speed: float | None = None
        follower_width = float(global_state.vehicle_width[world_idx, follower_idx].item())
        lateral_threshold = max(2.0, follower_width + self.leader_lateral_slack)

        for candidate_idx in vehicle_indices:
            if candidate_idx == follower_idx:
                continue
            progress, lateral_distance = self._project_progress(route, global_state, world_idx, candidate_idx)
            gap = progress - follower_progress
            if gap <= 0.0 or lateral_distance > lateral_threshold:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_speed = float(local_state.speed[world_idx, candidate_idx].item())
        return best_gap, best_speed

    def _project_progress(self, route: IDMVehicleRoute, global_state: Any, world_idx: int, agent_idx: int) -> tuple[float, float]:
        point = torch.stack((global_state.pos_x[world_idx, agent_idx], global_state.pos_y[world_idx, agent_idx]))
        start = route.route_xy[:-1]
        end = route.route_xy[1:]
        segment = end - start
        segment_len_sq = torch.clamp(torch.sum(segment * segment, dim=1), min=1e-6)
        alpha = torch.clamp(torch.sum((point[None, :] - start) * segment, dim=1) / segment_len_sq, 0.0, 1.0)
        projected = start + alpha[:, None] * segment
        distances = torch.linalg.vector_norm(point[None, :] - projected, dim=1)
        best_idx = int(torch.argmin(distances).item())
        segment_progress = route.route_progress[best_idx + 1] - route.route_progress[best_idx]
        progress = route.route_progress[best_idx] + alpha[best_idx] * segment_progress
        return float(progress.item()), float(distances[best_idx].item())

    def _interpolate_heading(self, route: IDMVehicleRoute, progress: float) -> float:
        query = torch.tensor(progress, dtype=route.route_progress.dtype, device=route.route_progress.device)
        idx = int(torch.searchsorted(route.route_progress, query, right=False).item())
        idx = min(max(idx, 0), route.route_heading.shape[0] - 1)
        return float(route.route_heading[idx].item())


class NavsimGpuDriveActor:
    """Composes an ego actor with NAVSIM-IDM vehicle actors into dense GPUDrive actions."""

    name = "navsim_gpudrive_actor"

    def __init__(self, ego_actor: ContinuousPolicyAgent, idm_actor: IDMActor) -> None:
        self.ego_actor = ego_actor
        self.idm_actor = idm_actor
        self.mappings: list[WorldActorMapping] = []

    @property
    def policy_name(self) -> str:
        return f"{self.ego_actor.name}+idm"

    def reset(self, env: Any, scene_jsons: Sequence[dict[str, Any]], sidecars: Sequence[dict[str, Any]]) -> None:
        from gpudrive.datatypes.metadata import Metadata
        from gpudrive.datatypes.observation import GlobalEgoState

        controlled_mask = env.cont_agent_mask
        metadata = Metadata.from_tensor(env.sim.metadata_tensor()).is_sdc.to(controlled_mask.device)
        global_state = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(), backend="torch", device=env.device)
        self.mappings = []

        for world_idx, (scene_json, sidecar) in enumerate(zip(scene_jsons, sidecars)):
            controlled_indices = torch.where(controlled_mask[world_idx])[0].tolist()
            ego_indices = [idx for idx in controlled_indices if int(metadata[world_idx, idx].item()) == 1]
            if len(ego_indices) != 1:
                raise ValueError(f"World {world_idx} expected exactly one controlled ego, got {ego_indices}")
            num_scene_objects = min(len(scene_json["objects"]), global_state.id.shape[1])
            id_to_index = {
                _agent_id(global_state.id[world_idx, agent_idx].item()): agent_idx
                for agent_idx in range(num_scene_objects)
            }
            idm_agent_ids = {_agent_id(vehicle["agent_id"]) for vehicle in sidecar.get("vehicles", {}).values()}
            expected_controlled = {_agent_id(global_state.id[world_idx, ego_indices[0]].item()), *idm_agent_ids}
            actual_controlled = {_agent_id(global_state.id[world_idx, idx].item()) for idx in controlled_indices}
            if actual_controlled != expected_controlled:
                raise ValueError(
                    f"World {world_idx} controlled ids mismatch: expected {sorted(expected_controlled)}, "
                    f"got {sorted(actual_controlled)}. Regenerate JSON with contiguous ids and ego/IDM-only mark_as_expert flags."
                )
            static_vehicle_indices = []
            for obj in scene_json["objects"]:
                obj_id = _agent_id(obj["id"])
                if obj["type"] == "vehicle" and not obj.get("is_sdc", False) and obj_id not in idm_agent_ids:
                    if obj_id in id_to_index:
                        static_vehicle_indices.append(id_to_index[obj_id])
            self.mappings.append(
                WorldActorMapping(
                    ego_index=ego_indices[0],
                    idm_indices=[id_to_index[agent_id] for agent_id in sorted(idm_agent_ids)],
                    static_vehicle_indices=static_vehicle_indices,
                    idm_agent_ids=idm_agent_ids,
                )
            )
        self.idm_actor.reset(sidecars, self.mappings, torch.device(env.device))

    def act(self, env: Any, obs: torch.Tensor, step_index: int) -> torch.Tensor:
        ego_actions = self.ego_actor.act(env, obs, step_index)
        actions = torch.zeros((env.num_worlds, env.max_agent_count, 3), dtype=torch.float32, device=env.device)
        for world_idx, mapping in enumerate(self.mappings):
            actions[world_idx, mapping.ego_index, :2] = ego_actions[world_idx]
        for (world_idx, agent_idx), (accel, steering) in self.idm_actor.act(env).items():
            actions[world_idx, agent_idx, 0] = accel
            actions[world_idx, agent_idx, 1] = steering
        if torch.any(~torch.isfinite(actions)):
            raise ValueError("Actor produced non-finite GPUDrive actions")
        actions[:, :, 0].clamp_(-6.0, 6.0)
        actions[:, :, 1].clamp_(-3.0, 3.0)
        actions[:, :, 2].zero_()
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
    parser.add_argument("--stage", choices=("stage_one", "stage_two", "all"), default="stage_one")
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ego-policy", choices=("expert_bicycle", "zero"), default="expert_bicycle")
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


def _sidecar_path(json_path: Path) -> Path:
    return json_path.with_suffix(".idm_routes.json")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_sidecar(path: Path) -> dict[str, Any]:
    sidecar_path = _sidecar_path(path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Missing IDM sidecar for {path}: {sidecar_path}")
    return _load_json(sidecar_path)


def _idm_vehicle_count(path: Path) -> int:
    return len(_load_sidecar(path).get("vehicles", {}))


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


def _make_env(paths: Sequence[Path], device: str, max_cont_agents: int):
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
        max_cont_agents=max_cont_agents,
        device=device,
        action_type="discrete",
    )


def _rollout_batch(paths: Sequence[Path], args: argparse.Namespace) -> list[dict[str, object]]:
    scene_jsons = [_load_json(path) for path in paths]
    sidecars = [_load_sidecar(path) for path in paths]
    idm_counts = [len(sidecar.get("vehicles", {})) for sidecar in sidecars]
    if len(set(idm_counts)) != 1:
        raise ValueError(f"Rollout batches must have equal idm vehicle counts, got {idm_counts}")

    env = _make_env(paths, args.device, max_cont_agents=1 + idm_counts[0])
    try:
        obs = env.reset(env.cont_agent_mask)
        controlled_mask = env.cont_agent_mask
        controlled_per_world = controlled_mask.sum(dim=1)
        expected_controlled = 1 + idm_counts[0]
        if not torch.all(controlled_per_world == expected_controlled):
            raise ValueError(f"Expected {expected_controlled} controlled agents per world, got {controlled_per_world.tolist()}")

        _, expert_positions, expert_velocities, expert_yaws = env.get_expert_actions()
        expert_actions = _infer_bicycle_actions(expert_velocities, expert_yaws)
        if args.ego_policy == "expert_bicycle":
            ego_policy: ContinuousPolicyAgent = ExpertPolicyAgent(expert_actions, controlled_mask)
        elif args.ego_policy == "zero":
            ego_policy = ZeroPolicyAgent()
        else:
            raise ValueError(f"Unknown ego policy: {args.ego_policy}")

        actor = NavsimGpuDriveActor(ego_policy, IDMActor())
        actor.reset(env, scene_jsons, sidecars)
        frames = [_sdc_global_poses(env)]

        for step_index in range(ROLLOUT_STEPS):
            action_tensor = actor.act(env, obs, step_index)
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
                    "policy": actor.policy_name,
                    "json_path": str(path),
                    "raw_path": str(raw_path),
                    "trajectory_path": str(trajectory_path),
                    "num_raw_states": ROLLOUT_STEPS + 1,
                    "num_navsim_poses": len(NAVSIM_SAMPLE_INDICES),
                    "num_controlled_agents": expected_controlled,
                    "num_idm_vehicles": idm_counts[world_index],
                    "num_static_vehicles": sidecars[world_index].get("static_vehicle_count", 0),
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
        "--stage",
        args.stage,
        "--json-dir",
        str(args.json_dir),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--ego-policy",
        args.ego_policy,
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
        "num_controlled_agents",
        "num_idm_vehicles",
        "num_static_vehicles",
        "max_position_error",
        "max_heading_error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _default_json_dir(stage: str) -> Path:
    return WORKSPACE_ROOT / "results/gpudrive_json/navhard_two_stage" / stage


def _default_output_dir(stage: str, ego_policy: str) -> Path:
    return WORKSPACE_ROOT / "results/gpudrive_rollouts/navhard_two_stage" / stage / f"{ego_policy}_idm"


def _stage_json_dir(path: Path | None, stage: str) -> Path:
    if path is None:
        return _default_json_dir(stage)
    if path.name == stage:
        return path
    return path / stage


def _stage_output_dir(path: Path | None, stage: str, ego_policy: str, all_stages: bool) -> Path:
    if path is None:
        return _default_output_dir(stage, ego_policy)
    if not all_stages:
        return path
    if path.name == stage:
        return path
    return path / stage / f"{ego_policy}_idm"


def _run_stage(args: argparse.Namespace, stage: str, json_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    files = _json_files(json_dir, args.max_scenes)
    rows: list[dict[str, object]] = []
    pending: list[Path] = []
    for path in files:
        token = _token_from_json(path)
        raw_path = output_dir / "raw" / f"{token}.npy"
        trajectory_path = output_dir / "trajectories" / f"{token}.npy"
        if raw_path.is_file() and trajectory_path.is_file() and not args.overwrite:
            raise FileExistsError(f"Rollout already exists for {token}; use --overwrite")
        pending.append(path)

    stage_args = argparse.Namespace(**vars(args))
    stage_args.stage = stage
    stage_args.json_dir = json_dir
    stage_args.output_dir = output_dir

    grouped: dict[int, list[Path]] = {}
    for path in pending:
        grouped.setdefault(_idm_vehicle_count(path), []).append(path)

    for idm_count in sorted(grouped):
        group = grouped[idm_count]
        for start in range(0, len(group), args.batch_size):
            batch = group[start : start + args.batch_size]
            rows.extend(_run_worker(batch, stage_args))
            _write_manifest(rows, output_dir)
            print(f"[{stage}] [{len(rows)}/{len(pending)}] completed (idm_vehicle_count={idm_count})", flush=True)

    if len(rows) != len(files):
        raise RuntimeError(f"Exported {len(rows)} trajectories for {len(files)} JSON files")
    print(f"Wrote {len(rows)} {stage} rollouts to {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.csv'}")
    return rows


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_position_error < 0 or args.max_heading_error < 0:
        raise ValueError("Error thresholds must be non-negative")

    if args.worker:
        if not args.worker_jsons or args.worker_result is None:
            raise ValueError("Worker mode requires --worker-jsons and --worker-result")
        rows = _rollout_batch([Path(path) for path in args.worker_jsons], args)
        with args.worker_result.open("w", encoding="utf-8") as file:
            json.dump(rows, file)
        return

    stages = ["stage_one", "stage_two"] if args.stage == "all" else [args.stage]
    total_rows = 0
    for stage in stages:
        json_dir = _stage_json_dir(args.json_dir, stage)
        output_dir = _stage_output_dir(args.output_dir, stage, args.ego_policy, args.stage == "all")
        total_rows += len(_run_stage(args, stage, json_dir, output_dir))
    if args.stage == "all":
        print(f"Wrote {total_rows} total rollouts across stage_one and stage_two")


if __name__ == "__main__":
    main()
