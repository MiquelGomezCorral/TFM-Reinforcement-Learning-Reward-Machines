import numpy as np


def compute_epsilon(min_epsilon, max_epsilon, step, decay_rate):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * step)


def compute_training_epsilon(
    min_epsilon,
    max_epsilon,
    step,
    decay_rate,
    learning_starts,
):
    """Keep exploration random during warm-up, then decay it exponentially."""
    steps = np.asarray(step)
    decay_steps = np.maximum(steps - learning_starts, 0)
    epsilon = compute_epsilon(min_epsilon, max_epsilon, decay_steps, decay_rate)
    epsilon = np.where(steps < learning_starts, 1.0, epsilon)
    return epsilon.item() if epsilon.ndim == 0 else epsilon


def should_optimize(total_steps, optimize_starts, optimize_interval):
    return total_steps >= optimize_starts and total_steps % optimize_interval == 0
