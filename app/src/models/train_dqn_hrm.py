import copy

from src.config import Configuration
from src.utils import seed_dqn
from .DQNHRM import DQNHRM
from .train_dqn import restore_best_checkpoint
from .train_hrm import train_hrm


def train_dqn_hrm(CONFIG: Configuration, agent: DQNHRM, get_propositions, env, progress_callback=None):
    if CONFIG.dqn_epsilon_decay_steps <= 0:
        raise ValueError("dqn_epsilon_decay_steps must be positive")
    if CONFIG.dqn_optimize_interval <= 0:
        raise ValueError("dqn_optimize_interval must be positive")
    if CONFIG.dqn_learning_starts < 0:
        raise ValueError("dqn_learning_starts cannot be negative")
    if CONFIG.dqn_checkpoint_interval <= 0:
        raise ValueError("dqn_checkpoint_interval must be positive")

    seed_dqn(CONFIG.seed)
    checkpoints = []

    def on_episode_end(episode, trained_agent, training_env, propositions):
        if episode % CONFIG.dqn_checkpoint_interval == 0:
            checkpoints.append((
                episode,
                copy.deepcopy(trained_agent.high_level.policy_net.state_dict()),
                copy.deepcopy(trained_agent.actor.policy_net.state_dict()),
            ))
        if progress_callback:
            progress_callback(episode, trained_agent, training_env, propositions)

    agent = train_hrm(CONFIG, agent, get_propositions, env, on_episode_end)
    restore_best_checkpoint(
        CONFIG,
        agent,
        get_propositions,
        env,
        checkpoints,
        (agent.high_level, agent.actor),
    )

    return agent
