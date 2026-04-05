import gymnasium as gym
import numpy as np
from tqdm import tqdm

from src.config import Configuration
from .QTable import QTable


def evaluate_agent(CONFIG: Configuration, qt: QTable, get_propositions: callable, env):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param max_steps: Maximum number of steps per episode
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param Q: The Q-table
  :param seed: The evaluation seed array (for taxi-v3)
  """
  env = env if env else gym.make(CONFIG.gym_id, render_mode="rgb_array")

  episode_rewards = []
  for episode in tqdm(range(CONFIG.n_eval_episodes)):
    if CONFIG.eval_seed:
      state, info = env.reset(seed=CONFIG.eval_seed[episode])
    else:
      state, info = env.reset()
    qt.reset_rm()

    terminated, truncated = False, False
    total_rewards_ep = 0

    for step in range(CONFIG.max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = qt.greedy_policy(state)
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward

      rm_done = False
      if qt.rm:
        events = get_propositions(env, new_state)
        _, _, rm_done = qt.step_rm(events)

      if terminated or truncated or rm_done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward