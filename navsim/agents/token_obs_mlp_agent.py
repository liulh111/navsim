import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """Dictionary with attribute access for model config values."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "ConfigDict":
        config = cls()
        for key, value in mapping.items():
            if isinstance(value, dict):
                value = cls.from_mapping(value)
            elif isinstance(value, list):
                value = [cls.from_mapping(item) if isinstance(item, dict) else item for item in value]
            config[key] = value
        return config


class TokenObsMLPAgent(AbstractAgent):
    """Agent that loads a token-named npy observation and predicts a trajectory with an external model."""

    requires_scene = False

    def __init__(
        self,
        obs_dir: str,
        model_module_path: str,
        model_class_name: str,
        model_config_path: str,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        :param obs_dir: Folder containing token-named npy files.
        :param model_module_path: Python file containing the model class.
        :param model_class_name: Class name to import from model_module_path.
        :param model_config_path: JSON file passed to the model constructor as a ConfigDict.
        :param trajectory_sampling: NAVSIM trajectory sampling used by PDM scoring.
        """
        super().__init__(trajectory_sampling)
        self._obs_dir = Path(obs_dir)
        self._model_module_path = Path(model_module_path)
        self._model_class_name = model_class_name
        self._model_config_path = Path(model_config_path)
        self._device = torch.device("cpu")
        self._model: Optional[torch.nn.Module] = None

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def initialize(self) -> None:
        """Import the external model class and initialize it from the JSON config."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cls = self._load_model_class()
        model_config = self._load_model_config()
        self._model = model_cls(model_config).to(self._device)
        self._model.eval()
        logger.info(
            "Initialized TokenObsMLPAgent model %s from %s with config %s",
            self._model_class_name,
            self._model_module_path,
            self._model_config_path,
        )

    def compute_trajectory(self, agent_input: AgentInput, token: str) -> Trajectory:
        """Load token observation and run the model."""
        if self._model is None:
            raise RuntimeError("TokenObsMLPAgent.initialize() must be called before compute_trajectory().")

        obs_path = self._obs_dir / f"{token}.npy"
        if not obs_path.is_file():
            raise FileNotFoundError(f"Observation npy file not found for token {token}: {obs_path}")

        obs = np.load(obs_path).astype(np.float32).reshape(-1)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            prediction = self._model(obs_tensor)

        poses = (
            prediction.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(self._trajectory_sampling.num_poses, 3)
        )
        return Trajectory(poses, self._trajectory_sampling)

    def _load_model_class(self) -> type:
        if not self._model_module_path.is_file():
            raise FileNotFoundError(f"MLP model module does not exist: {self._model_module_path}")

        spec = importlib.util.spec_from_file_location("token_obs_mlp_external_model", self._model_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import model module from {self._model_module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, self._model_class_name):
            raise AttributeError(f"Model class {self._model_class_name!r} not found in {self._model_module_path}")
        return getattr(module, self._model_class_name)

    def _load_model_config(self) -> ConfigDict:
        if not self._model_config_path.is_file():
            raise FileNotFoundError(f"Model config JSON does not exist: {self._model_config_path}")

        with self._model_config_path.open("r", encoding="utf-8") as config_file:
            model_config = json.load(config_file)

        if not isinstance(model_config, dict):
            raise ValueError(f"Model config JSON must contain an object: {self._model_config_path}")
        return ConfigDict.from_mapping(model_config)
