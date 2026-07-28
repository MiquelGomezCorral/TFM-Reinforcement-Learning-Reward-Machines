from tqdm import tqdm

from src.config import Configuration
from src.utils import compute_epsilon, next_episode_seed, seed_training
from .QTableHRM import QTableHRM


def _discounted_return(option_return, reward, gamma, option_steps) -> float:
    """Add the current RM reward to the discounted option return."""
    return option_return + gamma**option_steps * reward


def _high_level_target(
    discounted_return,
    option_steps,
    terminated,
    rm_done,
    gamma,
    qtable: QTableHRM,
    new_state,
    next_u,
) -> float:
    """Build the SMDP target for a completed high-level option."""
    if terminated or rm_done:
        return discounted_return

    return discounted_return + gamma ** (option_steps + 1) * qtable.max_high_value(
        new_state, next_u
    )


def train_qtable_hrm(
    CONFIG: Configuration,
    qtable: QTableHRM,
    get_propositions,
    env,
) -> QTableHRM:
    seed_generator = seed_training(CONFIG.seed, env.action_space)

    for episode in tqdm(range(CONFIG.n_training_episodes)):
        # Reduce epsilon because less exploration is needed over time.
        epsilon = compute_epsilon(
            CONFIG.min_epsilon,
            CONFIG.max_epsilon,
            episode,
            CONFIG.decay_rate,
        )

        # Reset the environment and RM at the start of each episode.
        observation, info = env.reset(seed=next_episode_seed(seed_generator))
        raw_state = info.get("raw_state", observation)
        qtable.reset_rm()
        state = CONFIG.parse_state(env, observation) if CONFIG.parse_state else observation

        option_start_state = None
        option_start_u = None
        option_return = 0
        option_steps = 0

        for step in range(CONFIG.max_steps):
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
                CONFIG.parse_state(env, new_observation)
                if CONFIG.parse_state else new_observation
            )
            events = get_propositions(env, raw_state, action, new_raw_state)
            next_u, reward, rm_done = qtable.step_rm(events)

            # Counterfactual updates
            qtable.counterfactual_update(events, terminated, state, action, new_state)

            # Update the high-level Q-table for the option that was active during this transition.
            discounted_return = _discounted_return(
                option_return, reward, CONFIG.gamma, option_steps
            )
            time_limit = step == CONFIG.max_steps - 1
            option_done = terminated or truncated or time_limit or next_u != current_u

            if option_done:
                high_target = _high_level_target(
                    discounted_return,
                    option_steps,
                    terminated,
                    rm_done,
                    CONFIG.gamma,
                    qtable,
                    new_state,
                    next_u,
                )
                qtable.high_level.update(
                    qtable.high_state(option_start_state, option_start_u),
                    qtable.target_action(active_option),
                    high_target,
                    qtable.high_state(new_state, next_u),
                    True,
                    CONFIG.gamma,
                    CONFIG.learning_rate,
                )
                qtable.active_option = None

            else:
                option_return = discounted_return
                option_steps += 1

            # Next iteration
            if terminated or truncated or rm_done:
                break

            state = new_state
            raw_state = new_raw_state

    return qtable
