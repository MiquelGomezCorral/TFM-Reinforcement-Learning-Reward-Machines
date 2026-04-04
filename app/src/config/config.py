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


    # ===================================================================
    #                       PARAMETER
    # ===================================================================

    exp_name: str = "base_name"
    seed:     int = 42

    gym_id:          str = None
    learning_rate: float = 2.5e-4
    total_timesteps: int = 25_000

    torch_deterministic: bool = True
    cuda:                bool = True

    track_run:         bool = False
    wandb_project_name: str = "RL"
    wandb_entity:       str = None

    def __post_init__(self):
        ...
        make_dirs([self.DATA_PATH, self.MODELS_PATH, self.LOGS_PATH])