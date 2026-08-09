from tqdm import tqdm

from src.config import Configuration
from src.utils import next_episode_seed, seed_training


def _discounted_return(option_return, reward, gamma, option_steps) -> float:
    """Add the current RM reward to the discounted option return."""
    return option_return + gamma**option_steps * reward


def _high_level_target(
    discounted_return,
    option_steps,
    terminated,
    rm_done,
    gamma,
    agent,
    new_state,
    next_u,
) -> float:
    """Build the SMDP target for a completed high-level option."""
    if terminated or rm_done:
        return discounted_return

    return discounted_return + agent.max_high_value(new_state, next_u) * gamma ** (option_steps + 1)


def train_hrm(
    CONFIG: Configuration,
    agent,
    get_propositions,
    env,
    progress_callback=None,
):
    seed_generator = seed_training(CONFIG.seed, env.action_space)
    total_steps = 0

    for episode in tqdm(range(CONFIG.n_training_episodes)):
        # Reset the environment and RM at the start of each episode.
        observation, info = env.reset(seed=next_episode_seed(seed_generator))
        raw_state = info.get("raw_state", observation)
        agent.reset_rm()
        state = observation

        option_start_high_state = None
        option_start_u = None
        option_return = 0
        option_steps = 0

        for step in range(CONFIG.max_steps):
            epsilon = agent.training_epsilon(episode, total_steps)
            current_u = agent.get_rm_state()
            if agent.active_option is None:
                option_start_high_state = agent.high_state(state, current_u)
                option_start_u = current_u
                agent.select_option(state, epsilon, current_u)
                option_return = 0
                option_steps = 0

            active_option = agent.active_option

            # The selected option controls primitive actions until the RM state changes.
            action = agent.epsilon_greedy_policy(state, epsilon, env.action_space.sample)
            new_observation, env_reward, terminated, truncated, info = env.step(action)
            new_raw_state = info.get("raw_state", new_observation)
            new_state = new_observation
            events = get_propositions(env, raw_state, action, new_raw_state)
            next_u, reward, rm_done = agent.step_rm(events)

            # Counterfactual updates
            agent.counterfactual_update(
                events,
                terminated,
                state,
                action,
                new_state,
                env_reward,
                info.get("invalid_action", False),
            )

            # Update the high-level policy for the option active during this transition.
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
                    agent,
                    new_state,
                    next_u,
                )
                agent.update_high_level(
                    option_start_high_state,
                    agent.target_action(active_option),
                    high_target,
                    new_state,
                    next_u,
                )
                agent.active_option = None

            else:
                option_return = discounted_return
                option_steps += 1

            agent.optimize_training_step(total_steps + 1)
            total_steps += 1

            # Next iteration
            if terminated or truncated or rm_done:
                break

            state = new_state
            raw_state = new_raw_state

        if progress_callback:
            progress_callback(episode + 1, agent, env, get_propositions)

    return agent
