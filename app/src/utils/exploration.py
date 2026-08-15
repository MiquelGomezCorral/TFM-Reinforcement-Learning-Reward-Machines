import numpy as np


def compute_epsilon(min_epsilon, max_epsilon, step, decay_rate):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * step)


def compute_training_epsilon(
    min_epsilon,
    max_epsilon,
    decay_step,
    decay_rate,
    warmup_step,
    learning_starts,
):
    epsilon = compute_epsilon(min_epsilon, max_epsilon, decay_step, decay_rate)
    if np.isscalar(warmup_step):
        return 1.0 if warmup_step < learning_starts else epsilon
    return np.where(np.asarray(warmup_step) < learning_starts, 1.0, epsilon)


def should_optimize(total_steps, optimize_starts, optimize_interval):
    return total_steps >= optimize_starts and total_steps % optimize_interval == 0
