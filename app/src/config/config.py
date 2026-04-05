"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass

from maikol_utils.file_utils import make_dirs

@dataclass 
class Configuration:
    """Configuration class for the project."""
    # ===================================================================
    #                       PATHS
    # ===================================================================
    DATA_PATH: str = os.path.join("..", "data")
    MODELS_PATH: str = os.path.join("..", "models")
    LOGS_PATH: str = os.path.join("..", "logs")
    VIDEO_PATH: str = os.path.join("..", "videos")


    # ===================================================================
    #                       PARAMETER
    # ===================================================================

    exp_name: str = "base_name"
    exp_description: str = "base_description"
    seed:     int = 42
    gym_id: str = "Taxi-v3"
    video_fps: int = 1

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

    # Exploration parameters
    max_epsilon: float = 1.0    # Exploration probability at start
    min_epsilon: float = 0.05   # Minimum exploration probability
    decay_rate: float = 0.0005  # Exponential decay rate for exploration prob

    def __post_init__(self):
        make_dirs([self.DATA_PATH, self.MODELS_PATH, self.LOGS_PATH, self.VIDEO_PATH])
        self.eval_seed = list(range(self.n_eval_episodes))