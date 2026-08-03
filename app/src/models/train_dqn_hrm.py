import copy
import random

import numpy as np
from tqdm import tqdm

from src.config import Configuration
from src.utils import compute_epsilon, next_episode_seed, seed_dqn, seed_training
from .DQNHRM import DQNHRM
from .HRM import option_reward
from .evaluate import evaluate_agent


def train_dqn_hrm(config: Configuration, agent: DQNHRM, get_propositions, env, progress_callback=None):
    if config.dqn_epsilon_decay_steps <= 0:
        raise ValueError("dqn_epsilon_decay_steps must be positive")
    if config.dqn_optimize_interval <= 0:
        raise ValueError("dqn_optimize_interval must be positive")
    if config.dqn_learning_starts < 0:
        raise ValueError("dqn_learning_starts cannot be negative")
    if config.dqn_checkpoint_interval <= 0:
        raise ValueError("dqn_checkpoint_interval must be positive")

    seed_dqn(config.seed)
    seed_generator = seed_training(config.seed, env.action_space)
    steps = 0
    checkpoints = []

    for episode in tqdm(range(config.n_training_episodes)):
        state, info = env.reset(seed=next_episode_seed(seed_generator))
        raw_state = info.get("raw_state", state)
        agent.reset_rm()

        option_start_state = None
        option_start_u = None
        option_return = 0
        option_steps = 0

        for step in range(config.max_steps):
            epsilon = compute_epsilon(
                config.min_epsilon,
                config.max_epsilon,
                steps,
                1 / config.dqn_epsilon_decay_steps,
            )
            current_u = agent.get_rm_state()
            if agent.active_option is None:
                option_start_state = np.array(state, dtype=np.float32, copy=True)
                option_start_u = current_u
                agent.select_option(state, epsilon, current_u)
                option_return = 0
                option_steps = 0

            active_option = agent.active_option
            action = agent.epsilon_greedy_policy(state, epsilon, env.action_space.sample)
            new_state, env_reward, terminated, truncated, info = env.step(action)
            new_raw_state = info.get("raw_state", new_state)
            events = get_propositions(env, raw_state, action, new_raw_state)
            next_u, reward, rm_done = agent.step_rm(events)
            time_limit = step == config.max_steps - 1

            reachable_states = set()
            for counterfactual_u, targets in agent.options.items():
                counterfactual_next_u, counterfactual_reward, _ = agent.rm.simulate_step(
                    counterfactual_u, events
                )
                reachable_states.add(counterfactual_next_u)
                if (
                    agent._valid_option_states is not None
                    and counterfactual_u not in agent._valid_option_states
                ):
                    continue
                option_done = terminated or counterfactual_next_u != counterfactual_u
                replay_terminal = option_done or truncated or time_limit
                for target_u in targets:
                    shaped_reward = option_reward(
                        env_reward if info.get("invalid_action", False) else counterfactual_reward,
                        target_u,
                        counterfactual_next_u,
                        option_done,
                        config.hrm_r_plus,
                        config.hrm_r_minus,
                    )
                    agent.actor.remember(
                        agent.actor_state(state, counterfactual_u, target_u),
                        action,
                        shaped_reward,
                        None if replay_terminal else agent.actor_state(
                            new_state, counterfactual_u, target_u
                        ),
                        replay_terminal,
                    )
            agent._valid_option_states = reachable_states

            discounted_return = option_return + config.gamma**option_steps * reward
            option_done = terminated or truncated or time_limit or next_u != current_u
            if option_done:
                high_target = discounted_return
                if not terminated and not rm_done:
                    high_target += config.gamma ** (option_steps + 1) * agent.max_high_value(
                        new_state, next_u
                    )
                # The SMDP bootstrap is already included in high_target.
                agent.high_level.update(
                    agent.high_state(option_start_state, option_start_u),
                    agent.target_action(active_option),
                    high_target,
                    None,
                    True,
                    optimize=False,
                )
                agent.active_option = None
            else:
                option_return = discounted_return
                option_steps += 1

            learning_started = steps + 1 >= config.dqn_learning_starts
            if (
                learning_started
                and (steps + 1) % config.dqn_optimize_interval == 0
            ):
                agent.actor.optimize()
                agent.high_level.optimize()

            steps += 1
            if terminated or truncated or rm_done:
                break
            state = new_state
            raw_state = new_raw_state

        if (episode + 1) % config.dqn_checkpoint_interval == 0:
            checkpoints.append((
                episode + 1,
                copy.deepcopy(agent.high_level.policy_net.state_dict()),
                copy.deepcopy(agent.actor.policy_net.state_dict()),
            ))
        if progress_callback:
            progress_callback(episode + 1, agent, env, get_propositions)

    if checkpoints:
        if checkpoints[-1][0] != config.n_training_episodes:
            checkpoints.append((
                config.n_training_episodes,
                copy.deepcopy(agent.high_level.policy_net.state_dict()),
                copy.deepcopy(agent.actor.policy_net.state_dict()),
            ))
        rng = random.Random(config.dqn_validation_seed_base)
        validation_seeds = [
            rng.randrange(2**32) for _ in range(config.dqn_validation_episodes)
        ]
        best_score = None
        best_episode = None
        best_high_state = None
        best_actor_state = None
        for checkpoint_episode, high_state, actor_state in checkpoints:
            agent.high_level.policy_net.load_state_dict(high_state)
            agent.actor.policy_net.load_state_dict(actor_state)
            metrics = evaluate_agent(
                config,
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
            f"validation_success={best_score[0]}/{config.dqn_validation_episodes}"
        )

    return agent
