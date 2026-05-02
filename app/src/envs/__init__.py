"""Environments.

Functions to manage, create, train / test environments.
"""

from .taxi import get_propositions_taxi
from .doorkey import get_propositions_doorkey, MiniGridDiscreteWrapper
from .taxi_big_env import MultiTaxiEnv