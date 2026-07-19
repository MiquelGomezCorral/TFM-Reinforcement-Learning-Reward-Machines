"""Environments.

Functions to manage, create, train / test environments.
"""

from .taxi import get_propositions_taxi, get_propositions_multi_taxi
from .doorkey import get_propositions_doorkey, MiniGridDiscreteWrapper
from .factory import create_environment
from .taxi_big_env import MultiTaxiEnv
