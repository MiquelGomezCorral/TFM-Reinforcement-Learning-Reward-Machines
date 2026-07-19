import numpy as np


def compute_epsilon(min_epsilon, max_epsilon, step, decay_rate):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * step)
