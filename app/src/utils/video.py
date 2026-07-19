import os
import imageio
import numpy as np

from src.config import Configuration
from src.models import QTableRM


def record_video(CONFIG: Configuration, qt: QTableRM, env, get_propositions: callable, video_name: str = None):
  """
  Record a replay of the greedy policy.

  :param CONFIG: Configuration object
  :param qt: Q-table or RM-indexed Q-table
  :param env: Environment to render
  :param get_propositions: Function that maps transitions to RM events
  :param video_name: Optional output filename
  """
  images = []
  raw_state, _ = env.reset(seed=CONFIG.seed)
  img = env.render()
  images.append(img)
  qt.reset_rm()
  state = CONFIG.parse_state(env, raw_state) if CONFIG.parse_state else raw_state

  for _ in range(CONFIG.max_steps):
    # Video replay is greedy: always take the action with max expected reward.
    action = qt.greedy_policy(state)
    new_state, _, terminated, truncated, _ = env.step(action)
    rm_done = False
    if qt.rm:
      _, _, rm_done = qt.step_rm(get_propositions(env, raw_state, action, new_state))

    img = env.render()
    images.append(img)

    if terminated or truncated or rm_done:
      break

    raw_state = new_state
    state = CONFIG.parse_state(env, raw_state) if CONFIG.parse_state else raw_state
  
  if video_name:
    path = os.path.join(CONFIG.VIDEO_PATH, video_name)
  else:
    path = os.path.join(CONFIG.VIDEO_PATH, f"{CONFIG.exp_description}_{CONFIG.gym_id}.gif")
    
  imageio.mimsave(path, [np.array(img) for img in images], fps=CONFIG.video_fps)
