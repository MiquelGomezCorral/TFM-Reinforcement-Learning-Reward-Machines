import os
import random 
import imageio
import numpy as np

from src.config import Configuration
from src.models import QTable


def record_video(CONFIG: Configuration, qt: QTable, env, get_propositions: callable, video_name: str = None):
  """
  Generate a replay video of the agent
  :param CONFIG: Configuration object
  :param Qtable: Qtable of our agent
  """
  images = []
  terminated, truncated = False, False
  state, info = env.reset(seed=random.randint(0,500))
  img = env.render()
  images.append(img)
  qt.reset_rm()
  if CONFIG.skip_first_rm_state:
      events = get_propositions(env, state)
      qt.step_rm(events)
  state = CONFIG.parse_state(env, state) if CONFIG.parse_state else state

  i = 0

  while not terminated and not truncated and i < CONFIG.max_steps:
    i+=1
    # Take the action (index) that have the maximum expected future reward given that state
    action = qt.greedy_policy(state)
    state, reward, terminated, truncated, info = env.step(action) # We directly put next_state = state for recording logic
    if CONFIG.use_rm:
      qt.step_rm(get_propositions(env, state))
    
    state = CONFIG.parse_state(env, state) if CONFIG.parse_state else state
    
    img = env.render()
    images.append(img)
  
  if video_name:
    path = os.path.join(CONFIG.VIDEO_PATH, video_name)
  else:
    path = os.path.join(CONFIG.VIDEO_PATH, f"{CONFIG.exp_description}_{CONFIG.gym_id}.gif")
    
  imageio.mimsave(path, [np.array(img) for i, img in enumerate(images)], fps=3)
