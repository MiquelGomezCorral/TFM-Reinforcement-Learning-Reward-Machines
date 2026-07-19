import numpy as np
from tqdm import tqdm

from src.config import Configuration
from .QTable import QTableRM


def evaluate_agent(CONFIG: Configuration, qt: QTableRM, get_propositions: callable, env):
  """
  Evaluate the greedy policy and return mean/std environment reward.

  :param CONFIG: Configuration object
  :param qt: Q-table or RM-indexed Q-table
  :param get_propositions: Function that maps transitions to RM events
  :param env: Evaluation environment
  """
  episode_rewards = []
  for episode in tqdm(range(CONFIG.n_eval_episodes)):
    if CONFIG.eval_seed:
      raw_state, _ = env.reset(seed=CONFIG.eval_seed[episode])
    else:
      raw_state, _ = env.reset()
    qt.reset_rm()

    total_rewards_ep = 0

    state = CONFIG.parse_state(env, raw_state) if CONFIG.parse_state else raw_state

    for _ in range(CONFIG.max_steps):
      # Evaluation is greedy: always take the action with max expected reward.
      action = qt.greedy_policy(state)
      new_state, reward, terminated, truncated, _ = env.step(action)
      total_rewards_ep += reward

      rm_done = False
      if qt.rm:
        events = get_propositions(env, raw_state, action, new_state)
        _, _, rm_done = qt.step_rm(events)

      if terminated or truncated or rm_done:
        break

      raw_state = new_state
      state = CONFIG.parse_state(env, raw_state) if CONFIG.parse_state else raw_state
      
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward
