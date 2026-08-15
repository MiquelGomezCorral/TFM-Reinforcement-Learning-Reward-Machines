"""Utils.

Utility functions for any tastk in the app.
"""
from .video import record_video
from .seeds import next_episode_seed, seed_dqn, seed_training
from .exploration import compute_epsilon, compute_training_epsilon, should_optimize
