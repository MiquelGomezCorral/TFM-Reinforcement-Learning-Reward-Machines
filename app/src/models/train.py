from typing import Callable
from tqdm import tqdm

from src.config import Configuration
from src.utils import compute_training_epsilon, next_episode_seed, seed_training
from .QTable import QTableRM


def train_qt(CONFIG: Configuration, qt: QTableRM, get_propositions: Callable, env, progress_callback=None):
  if CONFIG.qtable_learning_starts < 0:
    raise ValueError("qtable_learning_starts cannot be negative")

  seed_generator = seed_training(CONFIG.seed, env.action_space)
  
  for episode in tqdm(range(CONFIG.n_training_episodes)):
    seed = next_episode_seed(seed_generator)
    observation, info = env.reset(seed=seed)
    raw_state = info.get("raw_state", observation)
    qt.reset_rm()
    state = observation

    for _ in range(CONFIG.max_steps):
      action_epsilon = compute_training_epsilon(
        CONFIG.min_epsilon,
        CONFIG.max_epsilon,
        episode,
        CONFIG.decay_rate,
        CONFIG.qtable_learning_starts,
      )
      action = qt.epsilon_greedy_policy(state, action_epsilon, env)

      new_state, env_reward, terminated, truncated, info = env.step(action)
      new_raw_state = info.get("raw_state", new_state)

      rm_done = qt.update(
          state, action, env_reward, raw_state, new_raw_state, new_state,
          env,
          get_propositions,
          terminated=terminated,
          use_crm=CONFIG.use_crm,
        invalid_action=info.get("invalid_action", False),
      )
      if terminated or truncated or rm_done:
        break

      raw_state = new_raw_state
      state = new_state

    if progress_callback:
      progress_callback(episode + 1, qt, env, get_propositions)

  return qt
