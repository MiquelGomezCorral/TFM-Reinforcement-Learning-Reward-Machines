from typing import Callable
import numpy as np
from tqdm import tqdm

import gymnasium as gym
from src.config import Configuration
from .QTable import QTable

def train_qtable(CONFIG: Configuration, Qtable: QTable, get_propositions: Callable, env):
  env = env if env else gym.make(CONFIG.gym_id, render_mode="rgb_array")
  
  for episode in tqdm(range(CONFIG.n_training_episodes)):
    # Reduce epsilon (because we need less and less exploration)
    epsilon = CONFIG.min_epsilon + (CONFIG.max_epsilon - CONFIG.min_epsilon)*np.exp(-CONFIG.decay_rate*episode)
    # Reset the environment
    state, info = env.reset()
    Qtable.reset_rm()
    terminated, truncated= False, False

    if CONFIG.skip_first_rm_state:
      events = get_propositions(env, state)
      Qtable.step_rm(events)
    state = CONFIG.parse_state(env, state) if CONFIG.parse_state else state

    # Repeat
    for step in range(CONFIG.max_steps):
      # Choose the action At using epsilon greedy policy
      action = Qtable.epsilon_greedy_policy(state, epsilon, env)

      # Take action At and observe Rt+1 and St+1
      # Take the action (a) and observe the outcome state(s') and reward (r)
      new_state, env_reward, terminated, truncated, info = env.step(action)
      new_state_parse = CONFIG.parse_state(env, new_state) if CONFIG.parse_state else new_state

      # print(f"{state} --{action}--> {new_state_parse} | {new_state}, reward: {env_reward}")
      # Update Q(s,a)
      rm_done = Qtable.update(
          state, action, env_reward, new_state, new_state_parse, 
          CONFIG.gamma, CONFIG.learning_rate, 
          env, 
          get_propositions,
          use_crm=CONFIG.use_crm,  # Set to True when you want to enable CRM
          skip_first_rm_state=CONFIG.skip_first_rm_state
      )

      # If terminated or truncated finish the episode
      if terminated or truncated or rm_done:
        break

      # Our next state is the new state
      state = new_state_parse

  return Qtable
