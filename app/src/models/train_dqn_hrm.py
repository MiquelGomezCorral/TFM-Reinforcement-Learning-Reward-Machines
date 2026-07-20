import numpy as np
from tqdm import tqdm

from src.config import Configuration
from src.utils import compute_epsilon, next_episode_seed, seed_dqn, seed_training
from .DQNHRM import DQNHRM
from .HRM import option_reward


def train_dqn_hrm(config: Configuration, agent: DQNHRM, get_propositions, env):
    if config.dqn_epsilon_decay_steps <= 0:
        raise ValueError("dqn_epsilon_decay_steps must be positive")

    seed_dqn(config.seed)
    seed_generator = seed_training(config.seed, env.action_space)
    steps = 0

    for _ in tqdm(range(config.n_training_episodes)):
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
            new_state, _, terminated, truncated, info = env.step(action)
            new_raw_state = info.get("raw_state", new_state)
            events = get_propositions(env, raw_state, action, new_raw_state)
            next_u, reward, rm_done = agent.step_rm(events)

            for counterfactual_u, targets in agent.options.items():
                counterfactual_next_u, counterfactual_reward, _ = agent.rm.simulate_step(
                    counterfactual_u, events
                )
                option_done = terminated or counterfactual_next_u != counterfactual_u
                for target_u in targets:
                    shaped_reward = option_reward(
                        counterfactual_reward,
                        counterfactual_u,
                        target_u,
                        counterfactual_next_u,
                        config.hrm_r_plus,
                        config.hrm_r_minus,
                    )
                    agent.actor.remember(
                        agent.actor_state(state, counterfactual_u, target_u),
                        action,
                        shaped_reward,
                        None if option_done else agent.actor_state(
                            new_state, counterfactual_u, target_u
                        ),
                        option_done,
                    )
            agent.actor.optimize()

            discounted_return = option_return + config.gamma**option_steps * reward
            time_limit = step == config.max_steps - 1
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
                )
                agent.active_option = None
            else:
                option_return = discounted_return
                option_steps += 1

            steps += 1
            if terminated or truncated or rm_done:
                break
            state = new_state
            raw_state = new_raw_state

    return agent
