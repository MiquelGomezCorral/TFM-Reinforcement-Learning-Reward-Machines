import gymnasium as gym
import numpy as np


class OneHotDiscreteWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._num_states = env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._num_states,),
            dtype=np.float32,
        )

    def observation(self, observation):
        encoded = np.zeros(self._num_states, dtype=np.float32)
        encoded[observation] = 1.0
        return encoded

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = {**info, "raw_state": info.get("raw_state", observation)}
        return self.observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = {**info, "raw_state": info.get("raw_state", observation)}
        return self.observation(observation), reward, terminated, truncated, info
