import copy
import random

import numpy as np
from tqdm import tqdm

from src.config import Configuration
from src.utils import compute_training_epsilon, next_episode_seed, seed_dqn, seed_training, should_optimize
from .DQN import DQN
from .DQNRM import DQNRM
from .evaluate import evaluate_agent


def restore_best_checkpoint(CONFIG, agent, get_propositions, env, checkpoints, dqns):
    if not checkpoints:
        return

    if checkpoints[-1][0] != CONFIG.n_training_episodes:
        checkpoints.append((
            CONFIG.n_training_episodes,
            *(copy.deepcopy(dqn.policy_net.state_dict()) for dqn in dqns),
        ))
    rng = random.Random(CONFIG.dqn_validation_seed_base)
    validation_seeds = [
        rng.randrange(2**32) for _ in range(CONFIG.dqn_validation_episodes)
    ]
    best_score = None
    best_episode = None
    best_states = None
    for checkpoint_episode, *states in checkpoints:
        for dqn, state in zip(dqns, states):
            dqn.policy_net.load_state_dict(state)
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
            best_states = states
    for dqn, state in zip(dqns, best_states):
        dqn.policy_net.load_state_dict(state)
        dqn.target_net.load_state_dict(state)
    print(
        f" - Restored episode {best_episode} checkpoint: "
        f"validation_success={best_score[0]}/{CONFIG.dqn_validation_episodes}"
    )


def _train_vector_dqn(CONFIG, agent: DQN, get_propositions, env, evaluation_env, progress_callback):
    seed_generator = seed_training(CONFIG.seed, env.action_space)
    states, _ = env.reset(seed=[next_episode_seed(seed_generator) for _ in range(env.num_envs)])
    steps = 0
    episodes = 0

    while episodes < CONFIG.n_training_episodes:
        epsilons = compute_training_epsilon(
            CONFIG.min_epsilon,
            CONFIG.max_epsilon,
            np.arange(steps, steps + env.num_envs),
            1 / CONFIG.dqn_epsilon_decay_steps,
            CONFIG.dqn_learning_starts,
        )
        actions = agent.epsilon_greedy_policies(states, epsilons, env.action_space.sample)
        new_states, rewards, terminated, truncated, infos = env.step(actions)
        final_observations = infos.get("final_obs")
        if np.any(terminated | truncated) and final_observations is None:
            raise RuntimeError("Vector environment did not provide a final observation")

        for index in range(env.num_envs):
            done = terminated[index] or truncated[index]
            next_state = (
                final_observations[index]
                if done else new_states[index]
            )
            agent.remember(
                states[index],
                int(actions[index]),
                float(rewards[index]),
                next_state,
                bool(terminated[index]),
            )

        previous_steps = steps
        steps += env.num_envs
        for step in range(previous_steps + 1, steps + 1):
            if should_optimize(
                step, CONFIG.dqn_optimize_starts, CONFIG.dqn_optimize_interval
            ):
                agent.optimize()

        for _ in range(min(int(np.count_nonzero(terminated | truncated)), CONFIG.n_training_episodes - episodes)):
            episodes += 1
            if progress_callback:
                progress_callback(episodes, agent, evaluation_env, get_propositions)

        states = new_states

    return agent


def train_dqn(
    CONFIG: Configuration,
    agent: DQN | DQNRM,
    get_propositions,
    env,
    progress_callback=None,
    evaluation_env=None,
):
    if CONFIG.dqn_epsilon_decay_steps <= 0:
        raise ValueError("dqn_epsilon_decay_steps must be positive")
    if CONFIG.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    optimize_interval = CONFIG.dqn_optimize_interval
    if optimize_interval <= 0:
        raise ValueError("dqn_optimize_interval must be positive")
    if CONFIG.dqn_learning_starts < 0:
        raise ValueError("dqn_learning_starts cannot be negative")
    if CONFIG.dqn_optimize_starts < 0:
        raise ValueError("dqn_optimize_starts cannot be negative")
    if isinstance(agent, DQNRM) and CONFIG.dqn_checkpoint_interval <= 0:
        raise ValueError("dqn_checkpoint_interval must be positive")
    if CONFIG.dqn_num_envs <= 0:
        raise ValueError("dqn_num_envs must be positive")

    seed_dqn(CONFIG.seed)
    evaluation_env = env if evaluation_env is None else evaluation_env
    if CONFIG.dqn_num_envs > 1:
        if isinstance(agent, DQNRM):
            raise ValueError("Vectorized training currently supports plain DQN only")
        return _train_vector_dqn(
            CONFIG,
            agent,
            get_propositions,
            env,
            evaluation_env,
            progress_callback,
        )

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
            epsilon = compute_training_epsilon(
                CONFIG.min_epsilon,
                CONFIG.max_epsilon,
                steps,
                1 / CONFIG.dqn_epsilon_decay_steps,
                CONFIG.dqn_learning_starts,
            )
            action = agent.epsilon_greedy_policy(state, epsilon, env.action_space.sample)
            new_state, env_reward, terminated, truncated, info = env.step(action)
            new_raw_state = info.get("raw_state", new_state)
            optimize = should_optimize(
                steps + 1,
                CONFIG.dqn_optimize_starts,
                optimize_interval,
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

        if isinstance(agent, DQNRM) and (episode + 1) % CONFIG.dqn_checkpoint_interval == 0:
            checkpoints.append((episode + 1, copy.deepcopy(agent.dqn.policy_net.state_dict())))
        if progress_callback:
            progress_callback(episode + 1, agent, env, get_propositions)

    if isinstance(agent, DQNRM):
        restore_best_checkpoint(
            CONFIG, agent, get_propositions, evaluation_env, checkpoints, (agent.dqn,)
        )

    return agent
