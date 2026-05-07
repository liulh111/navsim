import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

logger = logging.getLogger(__name__)


class TokenObsMLPAgent(AbstractAgent):
    """Agent that loads a token-named npy observation and predicts a trajectory with an external MLP."""

    requires_scene = False

    def __init__(
        self,
        obs_dir: str,
        model_module_path: str,
        model_class_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_glob: str = "*.pt",
        obs_file_pattern: str = "{token}.npy",
        expected_obs_dim: int = 2984,
        trajectory_key: str = "trajectory",
        strict_load: bool = True,
        state_dict_prefixes_to_strip: Optional[List[str]] = None,
        device: str = "auto",
        trajectory_sampling: TrajectorySampling = TrajectorySampling(num_poses=40, interval_length=0.1),
    ):
        """
        :param obs_dir: Folder containing token-named npy files.
        :param model_module_path: Python file containing the MLP class.
        :param model_class_name: Class name to import from model_module_path.
        :param model_kwargs: Keyword arguments passed to the MLP constructor.
        :param checkpoint_path: Optional explicit checkpoint path.
        :param checkpoint_dir: Optional folder from which the latest matching checkpoint is loaded.
        :param checkpoint_glob: Glob used when checkpoint_dir is provided.
        :param obs_file_pattern: Pattern for obs files, defaults to "{token}.npy".
        :param expected_obs_dim: Flattened observation dimension.
        :param trajectory_key: Key used when the model returns a dict.
        :param strict_load: strict flag for torch.nn.Module.load_state_dict.
        :param state_dict_prefixes_to_strip: Prefixes stripped from checkpoint state dict keys.
        :param device: "auto", "cpu", or a torch device string.
        :param trajectory_sampling: NAVSIM trajectory sampling used by PDM scoring.
        """
        super().__init__(trajectory_sampling)
        self._obs_dir = Path(obs_dir)
        self._model_module_path = Path(model_module_path)
        self._model_class_name = model_class_name
        self._model_kwargs = model_kwargs or {}
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_glob = checkpoint_glob
        self._obs_file_pattern = obs_file_pattern
        self._expected_obs_dim = expected_obs_dim
        self._trajectory_key = trajectory_key
        self._strict_load = strict_load
        self._state_dict_prefixes_to_strip = state_dict_prefixes_to_strip or ["agent.", "model.", "module."]
        self._device_name = device
        self._device = torch.device("cpu")
        self._model: Optional[torch.nn.Module] = None

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def initialize(self) -> None:
        """Load the external MLP and its checkpoint."""
        if self._device_name == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self._device_name)

        model_cls = self._load_model_class()
        self._model = model_cls(**self._model_kwargs).to(self._device)
        checkpoint_path = self._resolve_checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_state_dict_prefixes(state_dict)
        self._model.load_state_dict(state_dict, strict=self._strict_load)
        self._model.eval()
        logger.info("Loaded TokenObsMLPAgent checkpoint from %s", checkpoint_path)

    def compute_trajectory(self, agent_input: AgentInput, token: str) -> Trajectory:
        """Load token observation and run the MLP."""
        if self._model is None:
            raise RuntimeError("TokenObsMLPAgent.initialize() must be called before compute_trajectory().")

        obs_path = self._obs_dir / self._obs_file_pattern.format(token=token)
        if not obs_path.is_file():
            raise FileNotFoundError(f"Observation npy file not found for token {token}: {obs_path}")

        obs = np.load(obs_path).astype(np.float32).reshape(-1)
        if self._expected_obs_dim is not None and obs.shape[0] != self._expected_obs_dim:
            raise ValueError(f"Expected obs dim {self._expected_obs_dim}, got {obs.shape[0]} for {obs_path}")

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            prediction = self._model(obs_tensor)

        poses = self._prediction_to_poses(prediction)
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

    def _resolve_checkpoint_path(self) -> Path:
        if self._checkpoint_path is not None:
            if not self._checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint path does not exist: {self._checkpoint_path}")
            return self._checkpoint_path

        if self._checkpoint_dir is None:
            raise ValueError("Either checkpoint_path or checkpoint_dir must be provided.")
        if not self._checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {self._checkpoint_dir}")

        checkpoint_paths = sorted(
            self._checkpoint_dir.glob(self._checkpoint_glob),
            key=lambda path: path.stat().st_mtime,
        )
        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No checkpoint matching {self._checkpoint_glob!r} found under {self._checkpoint_dir}"
            )
        return checkpoint_paths[-1]

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            if all(isinstance(key, str) for key in checkpoint.keys()):
                return checkpoint
        raise ValueError("Unsupported checkpoint format; expected a state_dict or a dict containing one.")

    def _strip_state_dict_prefixes(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stripped: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in self._state_dict_prefixes_to_strip:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            stripped[new_key] = value
        return stripped

    def _prediction_to_poses(self, prediction: Any) -> np.ndarray:
        if isinstance(prediction, dict):
            if self._trajectory_key not in prediction:
                raise KeyError(f"Model output dict does not contain trajectory key {self._trajectory_key!r}.")
            prediction = prediction[self._trajectory_key]
        elif isinstance(prediction, (tuple, list)):
            prediction = prediction[0]

        if isinstance(prediction, torch.Tensor):
            array = prediction.detach().cpu().numpy()
        else:
            array = np.asarray(prediction)

        array = array.astype(np.float32)
        if array.ndim == 3:
            array = array[0]
        elif array.ndim == 2 and array.shape[0] == 1:
            array = array.reshape(-1)

        expected_values = self._trajectory_sampling.num_poses * 3
        if array.shape == (self._trajectory_sampling.num_poses, 3):
            return array
        if array.size == expected_values:
            return array.reshape(self._trajectory_sampling.num_poses, 3)

        raise ValueError(
            "MLP output must be shaped "
            f"({self._trajectory_sampling.num_poses}, 3) or contain {expected_values} values, got {array.shape}."
        )
