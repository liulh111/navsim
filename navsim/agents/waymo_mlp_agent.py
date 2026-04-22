"""Agent that consumes the precomputed GPUDrive-derived 2984-dim Waymo feature
(`exp/pipeline_output/<TOKEN>/waymo_obs_gpudrive.npy`) and passes it through a
randomly initialised MLP to produce a navsim Trajectory.

The intent is a sanity check of the 3-stage pipeline: if `compute_trajectory`
can look up each scene's feature by token and produce a valid Trajectory, the
pipeline is wired correctly. No training / checkpoint loading is performed.
"""
from pathlib import Path

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory

WAYMO_FEATURE_DIM = 2984
DEFAULT_FEATURE_ROOT = "/data/llh/navsim_workspace/exp/pipeline_output"


class WaymoMLPAgent(AbstractAgent):
    """Random-init MLP that maps a 2984-dim Waymo observation to a Trajectory."""

    requires_scene = True

    def __init__(
        self,
        hidden_layer_dim: int = 256,
        feature_root: str = DEFAULT_FEATURE_ROOT,
        feature_filename: str = "waymo_obs_gpudrive.npy",
        seed: int = 0,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
        requires_scene: bool = True,
    ):
        super().__init__(trajectory_sampling, requires_scene)
        self._feature_root = Path(feature_root)
        self._feature_filename = feature_filename

        # Deterministic random init so repeated runs produce the same actions.
        prev_rng_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        try:
            num_poses = self._trajectory_sampling.num_poses
            self._mlp = torch.nn.Sequential(
                torch.nn.Linear(WAYMO_FEATURE_DIM, hidden_layer_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_layer_dim, num_poses * 3),
            )
        finally:
            torch.random.set_rng_state(prev_rng_state)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        # Nothing to load — weights are randomly initialised in __init__.
        self.eval()

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def _load_feature(self, token: str) -> torch.Tensor:
        path = self._feature_root / token / self._feature_filename
        if not path.is_file():
            raise FileNotFoundError(
                f"WaymoMLPAgent: missing precomputed feature for token={token}. "
                f"Expected at {path}. Run the 3-stage pipeline "
                f"(step1/step2/step3) for this token first."
            )
        arr = np.load(str(path)).astype(np.float32).reshape(-1)
        if arr.shape[0] != WAYMO_FEATURE_DIM:
            raise ValueError(
                f"WaymoMLPAgent: feature at {path} has shape {arr.shape}, "
                f"expected ({WAYMO_FEATURE_DIM},)."
            )
        return torch.from_numpy(arr)

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene, token: str) -> Trajectory:
        # agent_input is intentionally unused: the feature is precomputed on
        # disk, keyed by scene token, and already contains everything the MLP
        # needs. Signature must match AbstractAgent.
        del agent_input
        self.eval()
        feature = self._load_feature(token)  # (2984,)
        with torch.no_grad():
            pred = self._mlp(feature.unsqueeze(0))  # (1, num_poses * 3)
        num_poses = self._trajectory_sampling.num_poses
        poses = pred.view(num_poses, 3).cpu().numpy().astype(np.float32)
        return Trajectory(poses, self._trajectory_sampling)
