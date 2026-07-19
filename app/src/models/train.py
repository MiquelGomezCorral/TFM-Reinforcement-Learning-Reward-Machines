from typing import Callable
from tqdm import tqdm

from src.config import Configuration
from src.utils import compute_epsilon, next_episode_seed, seed_training
from .QTable import QTableRM

def train_qtable_crm(CONFIG: Configuration, Qtable: QTableRM, get_propositions: Callable, env):
  seed_generator = seed_training(CONFIG.seed, env.action_space)
  
  for episode in tqdm(range(CONFIG.n_training_episodes)):
    # Reduce epsilon because less exploration is needed over time.
    epsilon = compute_epsilon(
      CONFIG.min_epsilon, CONFIG.max_epsilon, episode, CONFIG.decay_rate
    )

    # Reset the environment and RM at the start of each episode.
    seed = next_episode_seed(seed_generator)
    raw_state, _ = env.reset(seed=seed)
    Qtable.reset_rm()

    state = CONFIG.parse_state(env, raw_state) if CONFIG.parse_state else raw_state

    for _ in range(CONFIG.max_steps):
      # Choose the action with epsilon-greedy exploration.
      action = Qtable.epsilon_greedy_policy(state, epsilon, env)

      # Take the action and observe S', reward, and terminal flags.
      new_state, env_reward, terminated, truncated, _ = env.step(action)
      new_state_parse = CONFIG.parse_state(env, new_state) if CONFIG.parse_state else new_state

      # Update Q(s,a), using RM reward if a reward machine is active.
      rm_done = Qtable.update(
          state, action, env_reward, raw_state, new_state, new_state_parse,
          CONFIG.gamma, CONFIG.learning_rate,
          env,
          get_propositions,
          terminated=terminated,
          use_crm=CONFIG.use_crm,
      )

      # Stop on true environment/RM terminal states or Gymnasium truncation.
      if terminated or truncated or rm_done:
        break

      raw_state = new_state
      state = new_state_parse

  return Qtable
