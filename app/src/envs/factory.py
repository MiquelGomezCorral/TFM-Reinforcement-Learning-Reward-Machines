from functools import partial

import gymnasium as gym
import minigrid  # Registers MiniGrid environments with Gymnasium.

from .doorkey import MiniGridDiscreteWrapper, get_propositions_doorkey
from .taxi import get_propositions_multi_taxi, get_propositions_taxi
from .taxi_big_env import MultiTaxiEnv
from .wrappers import OneHotDiscreteWrapper

from src.config import Configuration

ENVIRONMENT_FAMILIES = {
    "MiniGrid": (MiniGridDiscreteWrapper, get_propositions_doorkey),
    "Taxi": (None, get_propositions_taxi),
}


def create_environment(
    CONFIG: Configuration,
    one_hot_discrete: bool = False,
):
    family = CONFIG.gym_id.split("-", 1)[0]
    if family == "MultiTaxi":
        env = MultiTaxiEnv(
            grid_size=CONFIG.multitaxi_grid_size,
            num_passengers=CONFIG.multitaxi_num_passengers,
            observation_mode=CONFIG.multitaxi_observation_mode,
            render_mode="rgb_array",
            reward_shaping=CONFIG.multitaxi_reward_shaping,
            non_terminal_reward=CONFIG.multitaxi_non_terminal_reward,
        )
        get_propositions = get_propositions_multi_taxi
    else:
        if family not in ENVIRONMENT_FAMILIES:
            raise ValueError(f"Unsupported gym_id: {CONFIG.gym_id}")

        wrapper, get_propositions = ENVIRONMENT_FAMILIES[family]
        env = gym.make(CONFIG.gym_id, render_mode="rgb_array")
        if wrapper:
            env = wrapper(env)

    if one_hot_discrete and isinstance(env.observation_space, gym.spaces.Discrete):
        env = OneHotDiscreteWrapper(env)

    return env, get_propositions


def _create_time_limited_environment(CONFIG, one_hot_discrete):
    env, _ = create_environment(CONFIG, one_hot_discrete)
    return gym.wrappers.TimeLimit(env, max_episode_steps=CONFIG.max_steps)


def create_vector_environment(CONFIG: Configuration, num_envs: int, one_hot_discrete: bool = False):
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    return gym.vector.AsyncVectorEnv(
        [partial(_create_time_limited_environment, CONFIG, one_hot_discrete) for _ in range(num_envs)],
        context="spawn",
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
