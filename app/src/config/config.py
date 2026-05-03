"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass
from typing import Callable

from maikol_utils.file_utils import make_dirs
import yaml

@dataclass 
class Configuration:
    """Configuration class for the project."""
    # ===================================================================
    #                       PATHS
    # ===================================================================
    CONFIGS_PATH: str = os.path.join("..", "configs")
    DATA_PATH: str = os.path.join("..", "data")
    MODELS_PATH: str = os.path.join("..", "models")
    LOGS_PATH: str = os.path.join("..", "logs")
    VIDEO_PATH: str = os.path.join("..", "videos")
    yaml_config_path: str = None

    # ===================================================================
    #                       PARAMETER
    # ===================================================================

    exp_name: str = "base_name"
    rm_file: str = "rm_taxi.txt"
    exp_description: str = "base_description"
    seed:     int = 42
    gym_id: str = "Taxi-v3"
    video_fps: int = 10

    # Training parameters
    n_training_episodes: int = 10000  # Total training episodes
    learning_rate: float = 0.7  # Learning rate

    # Evaluation parameters
    n_eval_episodes: int = 100  # Total number of test episodes

    # Environment parameters
    max_steps: int = 99         # Max steps per episode
    gamma: float = 0.95         # Discounting rate
    eval_seed: list = None      # The evaluation seed of the environment
    use_rm: bool = False        # Whether to use the RM or not
    use_crm: bool = False       # Whether to use the CRM or not
    skip_first_rm_state: bool = False
    parse_state: Callable = None
    dynamic_qtable: bool = True

    # Exploration parameters
    max_epsilon: float = 1.0    # Exploration probability at start
    min_epsilon: float = 0.05   # Minimum exploration probability
    decay_rate: float = 0.0005  # Exponential decay rate for exploration prob

    def _load_yaml_configuration(self, yaml_file: str) -> None:
        """Load config values from a YAML file under CONFIGS_PATH."""
        config_path = os.path.join(self.CONFIGS_PATH, yaml_file)

        with open(config_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file) or {}

        for key, value in yaml_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __post_init__(self):
        make_dirs([self.DATA_PATH, self.MODELS_PATH, self.LOGS_PATH, self.VIDEO_PATH, self.CONFIGS_PATH])

        if self.yaml_config_path:
            self._load_yaml_configuration(self.yaml_config_path)

        self.eval_seed = list(range(self.n_eval_episodes))