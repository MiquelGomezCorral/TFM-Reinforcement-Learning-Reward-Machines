import gymnasium as gym
import minigrid  # Registers MiniGrid environments with Gymnasium.

from .doorkey import MiniGridDiscreteWrapper, get_propositions_doorkey
from .taxi import get_propositions_multi_taxi, get_propositions_taxi
from .taxi_big_env import MultiTaxiEnv
from .wrappers import OneHotDiscreteWrapper


ENVIRONMENT_FAMILIES = {
    "MiniGrid": (MiniGridDiscreteWrapper, get_propositions_doorkey),
    "Taxi": (None, get_propositions_taxi),
}


def create_environment(
    gym_id: str,
    one_hot_discrete: bool = False,
    multitaxi_grid_size: int = 5,
    multitaxi_num_passengers: int = 2,
    multitaxi_observation_mode: str = "discrete",
):
    family = gym_id.split("-", 1)[0]
    if family == "MultiTaxi":
        env = MultiTaxiEnv(
            grid_size=multitaxi_grid_size,
            num_passengers=multitaxi_num_passengers,
            observation_mode=multitaxi_observation_mode,
            render_mode="rgb_array",
        )
        get_propositions = get_propositions_multi_taxi
    else:
        if family not in ENVIRONMENT_FAMILIES:
            raise ValueError(f"Unsupported gym_id: {gym_id}")

        wrapper, get_propositions = ENVIRONMENT_FAMILIES[family]
        env = gym.make(gym_id, render_mode="rgb_array")
        if wrapper:
            env = wrapper(env)

    if one_hot_discrete and isinstance(env.observation_space, gym.spaces.Discrete):
        env = OneHotDiscreteWrapper(env)

    return env, get_propositions
