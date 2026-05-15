import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch


class ConfigDict(dict):
    """Dictionary with attribute access for local config generation."""

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


class SimpleTokenObsMLP(torch.nn.Module):
    """Small MLP that maps one flattened token observation to 8 future poses."""

    def __init__(self, config: ConfigDict, load_weights: bool = True):
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        num_layers = int(config.num_layers)
        input_dim = int(config.input_dim)
        output_dim = int(config.num_poses) * 3

        layers = []
        layer_input_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    torch.nn.Linear(layer_input_dim, hidden_dim),
                    torch.nn.ReLU(),
                ]
            )
            layer_input_dim = hidden_dim
        layers.append(torch.nn.Linear(layer_input_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)
        self.num_poses = int(config.num_poses)

        if load_weights:
            state_dict = torch.load(config.model_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs.float())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a simple token-observation MLP checkpoint and config.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/token_obs_mlp_demo"),
        help="Directory where simple_mlp.pt and config.json are written.",
    )
    parser.add_argument("--input-dim", type=int, default=2984)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-poses", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "simple_mlp.pt"
    config_path = output_dir / "config.json"
    config = ConfigDict.from_mapping(
        {
            "model_path": str(model_path),
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_poses": args.num_poses,
        }
    )

    torch.manual_seed(args.seed)
    model = SimpleTokenObsMLP(config, load_weights=False)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    torch.save(model.state_dict(), config.model_path)
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(dict(config), config_file, indent=2)
        config_file.write("\n")

    print(f"Saved model to {config.model_path}")
    print(f"Saved config to {config_path}")
