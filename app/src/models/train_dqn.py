import copy
import random

from tqdm import tqdm

from src.config import Configuration
from src.utils import compute_epsilon, next_episode_seed, seed_dqn, seed_training
from .DQN import DQN
from .DQNRM import DQNRM
from .evaluate import evaluate_agent


def train_dqn(CONFIG: Configuration, agent: DQN | DQNRM, get_propositions, env, progress_callback=None):
    if CONFIG.dqn_epsilon_decay_steps <= 0:
        raise ValueError("dqn_epsilon_decay_steps must be positive")
    if CONFIG.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    optimize_interval = CONFIG.dqn_optimize_interval
    if optimize_interval <= 0:
        raise ValueError("dqn_optimize_interval must be positive")
    if CONFIG.dqn_learning_starts < 0:
        raise ValueError("dqn_learning_starts cannot be negative")
    if CONFIG.dqn_checkpoint_interval <= 0:
        raise ValueError("dqn_checkpoint_interval must be positive")

    seed_dqn(CONFIG.seed)
    seed_generator = seed_training(CONFIG.seed, env.action_space)
    steps = 0
    checkpoints = []

    for episode in tqdm(range(CONFIG.n_training_episodes)):
        seed = next_episode_seed(seed_generator)
        state, info = env.reset(seed=seed)
        raw_state = info.get("raw_state", state)
        if isinstance(agent, DQNRM):
            agent.reset_rm()

        for _ in range(CONFIG.max_steps):
            epsilon = compute_epsilon(
                CONFIG.min_epsilon,
                CONFIG.max_epsilon,
                steps,
                1 / CONFIG.dqn_epsilon_decay_steps,
            )
            action = agent.epsilon_greedy_policy(state, epsilon, env.action_space.sample)
            new_state, env_reward, terminated, truncated, info = env.step(action)
            new_raw_state = info.get("raw_state", new_state)
            optimize = (
                steps + 1 >= CONFIG.dqn_learning_starts
                and (steps + 1) % optimize_interval == 0
            )

            if isinstance(agent, DQNRM):
                rm_done = agent.update(
                    state,
                    action,
                    env_reward,
                    raw_state,
                    new_raw_state,
                    new_state,
                    terminated,
                    env,
                    get_propositions,
                    use_crm=CONFIG.use_crm,
                    optimize=optimize,
                )
            else:
                agent.update(state, action, env_reward, new_state, terminated, optimize=optimize)
                rm_done = False

            steps += 1
            if terminated or truncated or rm_done:
                break
            state = new_state
            raw_state = new_raw_state

        if (episode + 1) % CONFIG.dqn_checkpoint_interval == 0:
            dqn = agent.dqn if isinstance(agent, DQNRM) else agent
            checkpoints.append((episode + 1, copy.deepcopy(dqn.policy_net.state_dict())))
        if progress_callback:
            progress_callback(episode + 1, agent, env, get_propositions)

    if checkpoints:
        dqn = agent.dqn if isinstance(agent, DQNRM) else agent
        if checkpoints[-1][0] != CONFIG.n_training_episodes:
            checkpoints.append((
                CONFIG.n_training_episodes,
                copy.deepcopy(dqn.policy_net.state_dict()),
            ))
        rng = random.Random(CONFIG.dqn_validation_seed_base)
        validation_seeds = [
            rng.randrange(2**32) for _ in range(CONFIG.dqn_validation_episodes)
        ]
        best_score = None
        best_episode = None
        best_policy_state = None
        for checkpoint_episode, policy_state in checkpoints:
            dqn.policy_net.load_state_dict(policy_state)
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
                best_policy_state = policy_state
        dqn.policy_net.load_state_dict(best_policy_state)
        dqn.target_net.load_state_dict(best_policy_state)
        print(
            f" - Restored episode {best_episode} checkpoint: "
            f"validation_success={best_score[0]}/{CONFIG.dqn_validation_episodes}"
        )

    return agent
