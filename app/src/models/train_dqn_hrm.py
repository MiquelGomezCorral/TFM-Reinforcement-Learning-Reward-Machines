import copy
import random

from src.config import Configuration
from src.utils import seed_dqn
from .DQNHRM import DQNHRM
from .evaluate import evaluate_agent
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

    def checkpoint(episode, trained_agent, _env, _get_propositions):
        if episode % CONFIG.dqn_checkpoint_interval == 0:
            checkpoints.append((
                episode,
                copy.deepcopy(trained_agent.high_level.policy_net.state_dict()),
                copy.deepcopy(trained_agent.actor.policy_net.state_dict()),
            ))
        if progress_callback:
            progress_callback(episode, trained_agent, _env, _get_propositions)

    agent = train_hrm(CONFIG, agent, get_propositions, env, checkpoint)

    if checkpoints:
        if checkpoints[-1][0] != CONFIG.n_training_episodes:
            checkpoints.append((
                CONFIG.n_training_episodes,
                copy.deepcopy(agent.high_level.policy_net.state_dict()),
                copy.deepcopy(agent.actor.policy_net.state_dict()),
            ))
        rng = random.Random(CONFIG.dqn_validation_seed_base)
        validation_seeds = [
            rng.randrange(2**32) for _ in range(CONFIG.dqn_validation_episodes)
        ]
        best_score = None
        best_episode = None
        best_high_state = None
        best_actor_state = None
        for checkpoint_episode, high_state, actor_state in checkpoints:
            agent.high_level.policy_net.load_state_dict(high_state)
            agent.actor.policy_net.load_state_dict(actor_state)
            metrics = evaluate_agent(
                CONFIG,
                agent,
                get_propositions,
                env,
                seeds=validation_seeds,
                report=False,
                return_metrics=True,
            )
            score = (
                metrics["successes"],
                -metrics["invalid_actions"],
                metrics["mean_reward"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_episode = checkpoint_episode
                best_high_state = high_state
                best_actor_state = actor_state
        agent.high_level.policy_net.load_state_dict(best_high_state)
        agent.high_level.target_net.load_state_dict(best_high_state)
        agent.actor.policy_net.load_state_dict(best_actor_state)
        agent.actor.target_net.load_state_dict(best_actor_state)
        print(
            f" - Restored episode {best_episode} checkpoint: "
            f"validation_success={best_score[0]}/{CONFIG.dqn_validation_episodes}"
        )

    return agent
