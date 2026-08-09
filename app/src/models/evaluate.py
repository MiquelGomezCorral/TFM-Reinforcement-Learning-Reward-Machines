import numpy as np
from tqdm import tqdm

from src.config import Configuration


def evaluate_agent(
  CONFIG: Configuration,
  qt,
  get_propositions: callable,
  env,
  seeds=None,
  report=True,
  return_metrics=False,
):
  """
  Evaluate the greedy policy and return mean/std environment reward.

  :param CONFIG: Configuration object
  :param qt: Q-table or RM-indexed Q-table
  :param get_propositions: Function that maps transitions to RM events
  :param env: Evaluation environment
  """
  episode_rewards = []
  successful_rewards = []
  successful_steps = []
  invalid_actions = 0
  evaluation_seeds = CONFIG.eval_seed if seeds is None else seeds
  episode_count = len(evaluation_seeds) if evaluation_seeds else CONFIG.n_eval_episodes
  for episode in tqdm(range(episode_count), disable=not report):
    if evaluation_seeds:
      state, info = env.reset(seed=evaluation_seeds[episode])
    else:
      state, info = env.reset()
    raw_state = info.get("raw_state", state)
    if hasattr(qt, "reset_rm"):
      qt.reset_rm()

    total_rewards_ep = 0

    completed = False
    for step in range(CONFIG.max_steps):
      # Evaluation is greedy: always take the action with max expected reward.
      action = qt.greedy_policy(state)
      new_state, reward, terminated, truncated, info = env.step(action)
      new_raw_state = info.get("raw_state", new_state)
      total_rewards_ep += reward
      invalid_actions += int(info.get("invalid_action", False))

      rm_done = False
      if getattr(qt, "rm", None):
        events = get_propositions(env, raw_state, action, new_raw_state)
        _, _, rm_done = qt.step_rm(events)

      if terminated or truncated or rm_done:
        completed = terminated or rm_done
        break

      raw_state = new_raw_state
      state = new_state
      
    episode_rewards.append(total_rewards_ep)
    if completed:
      successful_rewards.append(total_rewards_ep)
      successful_steps.append(step + 1)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  successful_std = np.std(successful_rewards) if successful_rewards else float("nan")
  mean_successful_steps = np.mean(successful_steps) if successful_steps else float("nan")
  metrics = {
    "successes": len(successful_rewards),
    "episodes": episode_count,
    "invalid_actions": invalid_actions,
    "mean_reward": mean_reward,
    "reward_std": std_reward,
    "successful_std": successful_std,
    "mean_successful_steps": mean_successful_steps,
    "worst_reward": min(episode_rewards),
  }
  if report:
    print(
      f" - Success={metrics['successes']}/{episode_count}, "
      f"timeouts={episode_count - metrics['successes']}, "
      f"invalid_actions={invalid_actions}, successful_std={successful_std:.2f}, "
      f"mean_successful_steps={mean_successful_steps:.2f}, "
      f"worst_reward={metrics['worst_reward']:.0f}"
    )

  return metrics if return_metrics else (mean_reward, std_reward)
