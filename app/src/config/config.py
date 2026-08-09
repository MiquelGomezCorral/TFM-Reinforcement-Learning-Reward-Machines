"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
import random
from dataclasses import dataclass

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
    rm_file: str = "rm_taxi_2p.txt"
    exp_description: str = "base_description"
    seed:     int = 42
    gym_id: str = "MultiTaxi-v0"
    video_fps: int = 10

    # Training parameters
    n_training_episodes: int = 10000  # Total training episodes
    learning_rate: float = 0.7  # Learning rate

    # Evaluation parameters
    n_eval_episodes: int = 100  # Total number of test episodes

    # Environment parameters
    max_steps: int = 99         # Max steps per episode
    gamma: float = 0.95         # Discounting rate
    eval_seed: list | None = None
    eval_seed_base: int | None = None
    use_rm: bool = False        # Whether to use the RM or not
    use_crm: bool = False       # Whether to use the CRM or not
    dynamic_qtable: bool = True
    multitaxi_grid_size: int = 5
    multitaxi_num_passengers: int = 2
    multitaxi_observation_mode: str = "discrete"

    # Exploration parameters
    max_epsilon: float = 1.0    # Exploration probability at start
    min_epsilon: float = 0.05   # Minimum exploration probability
    decay_rate: float = 0.0005  # Exponential decay rate for exploration prob

    # HRM parameters
    hrm_r_plus: float = 1
    hrm_r_minus: float = 0
    hrm_q_init: float = 0       # Use 2 for the paper's optimistic initialization

    # DQN parameters
    dqn_batch_size: int = 128
    dqn_replay_capacity: int = 10000
    dqn_learning_rate: float = 1e-3
    dqn_hidden_size: int = 128
    dqn_tau: float = 0.005
    dqn_gradient_clip: float = 100
    dqn_epsilon_decay_steps: int = 2500
    dqn_optimize_interval: int = 4
    dqn_learning_starts: int = 0
    dqn_num_envs: int = 1
    dqn_checkpoint_interval: int = 5000
    dqn_validation_episodes: int = 200
    dqn_validation_seed_base: int = 20260719

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

        if self.eval_seed is None:
            self.generate_eval_seeds()

    def generate_eval_seeds(self) -> None:
        rng = random.Random(self.seed if self.eval_seed_base is None else self.eval_seed_base)
        self.eval_seed = [rng.randrange(2**32) for _ in range(self.n_eval_episodes)]

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.generate_eval_seeds()
