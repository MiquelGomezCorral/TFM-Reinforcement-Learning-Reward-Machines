"""Models.

Functions to manage, create, train / test models.
"""

from .RewardMachine import RewardMachine
from .QTable import QTable, QTableRM
from .train import train_qtable_crm
from .evaluate import evaluate_agent
