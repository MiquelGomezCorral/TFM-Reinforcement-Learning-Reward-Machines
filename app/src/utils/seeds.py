import random

import numpy as np


def seed_training(seed, action_space):
    np.random.seed(seed)
    action_space.seed(seed)
    return np.random.default_rng(seed)


def seed_dqn(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_episode_seed(seed_generator):
    return int(seed_generator.integers(0, 2**31))
