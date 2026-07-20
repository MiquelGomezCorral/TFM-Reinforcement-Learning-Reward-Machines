import numpy as np
from tqdm import tqdm

from src.config import Configuration
from .QTableHRM import QTableHRM


def _option_reward(base_reward, u, target_u, next_u, r_plus, r_minus):
    if target_u == u:
        return base_reward
    return base_reward + (r_plus if next_u == target_u else r_minus)


def train_qtable_hrm(
    config: Configuration,
    qtable: QTableHRM,
    get_propositions,
    env,
):
    seed_generator = np.random.default_rng(config.seed)
    np.random.seed(config.seed)
    env.action_space.seed(config.seed)

    for episode in tqdm(range(config.n_training_episodes)):
        # Reduce epsilon because less exploration is needed over time.
        epsilon = config.min_epsilon + (
            config.max_epsilon - config.min_epsilon
        ) * np.exp(-config.decay_rate * episode)

        # Reset the environment and RM at the start of each episode.
        observation, info = env.reset(seed=int(seed_generator.integers(0, 2**31)))
        raw_state = info.get("raw_state", observation)
        qtable.reset_rm()
        state = config.parse_state(env, observation) if config.parse_state else observation

        option_start_state = None
        option_start_u = None
        option_return = 0
        option_steps = 0

        for step in range(config.max_steps):
            current_u = qtable.get_rm_state()
            if qtable.active_option is None:
                option_start_state = state
                option_start_u = current_u
                qtable.select_option(state, epsilon, current_u)
                option_return = 0
                option_steps = 0

            active_option = qtable.active_option

            # The selected option controls primitive actions until the RM state changes.
            action = qtable.epsilon_greedy_policy(state, epsilon, env)
            new_observation, _, terminated, truncated, info = env.step(action)
            new_raw_state = info.get("raw_state", new_observation)
            new_state = (
                config.parse_state(env, new_observation)
                if config.parse_state else new_observation
            )
            events = get_propositions(env, raw_state, action, new_raw_state)
            next_u, reward, rm_done = qtable.step_rm(events)

            # Every option learns from the transition as a counterfactual experience.
            for counterfactual_u, targets in qtable.options.items():
                counterfactual_next_u, counterfactual_reward, _ = qtable.rm.simulate_step(
                    counterfactual_u, events
                )
                option_done = terminated or counterfactual_next_u != counterfactual_u
                for target_u in targets:
                    shaped_reward = _option_reward(
                        counterfactual_reward,
                        counterfactual_u,
                        target_u,
                        counterfactual_next_u,
                        config.hrm_r_plus,
                        config.hrm_r_minus,
                    )
                    qtable.actor.update(
                        qtable.actor_state(state, counterfactual_u, target_u),
                        action,
                        shaped_reward,
                        qtable.actor_state(new_state, counterfactual_u, target_u),
                        option_done,
                        config.gamma,
                        config.learning_rate,
                    )

            discounted_return = option_return + config.gamma**option_steps * reward
            time_limit = step == config.max_steps - 1
            option_done = terminated or truncated or time_limit or next_u != current_u
            if option_done:
                high_target = discounted_return
                if not terminated and not rm_done:
                    high_target += config.gamma ** (option_steps + 1) * qtable.max_high_value(
                        new_state, next_u
                    )
                qtable.high_level.update(
                    qtable.high_state(option_start_state, option_start_u),
                    qtable.target_action(active_option),
                    high_target,
                    qtable.high_state(new_state, next_u),
                    True,
                    config.gamma,
                    config.learning_rate,
                )
                qtable.active_option = None
            else:
                option_return = discounted_return
                option_steps += 1

            if terminated or truncated or rm_done:
                break

            state = new_state
            raw_state = new_raw_state

    return qtable
