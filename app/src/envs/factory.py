import gymnasium as gym
import minigrid  # Registers MiniGrid environments with Gymnasium.

from .doorkey import MiniGridDiscreteWrapper, get_propositions_doorkey
from .taxi import get_propositions_taxi


ENVIRONMENT_FAMILIES = {
    "MiniGrid": (MiniGridDiscreteWrapper, get_propositions_doorkey),
    "Taxi": (None, get_propositions_taxi),
}


def create_environment(gym_id: str):
    family = gym_id.split("-", 1)[0]
    if family not in ENVIRONMENT_FAMILIES:
        raise ValueError(f"Unsupported gym_id: {gym_id}")

    wrapper, get_propositions = ENVIRONMENT_FAMILIES[family]
    env = gym.make(gym_id, render_mode="rgb_array")
    if wrapper:
        env = wrapper(env)

    return env, get_propositions
