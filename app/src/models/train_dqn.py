from tqdm import tqdm

from src.config import Configuration
from src.utils import compute_epsilon, next_episode_seed, seed_dqn, seed_training
from .DQN import DQN
from .DQNRM import DQNRM


def train_dqn(config: Configuration, agent: DQN | DQNRM, get_propositions, env):
    if config.dqn_epsilon_decay_steps <= 0:
        raise ValueError("dqn_epsilon_decay_steps must be positive")

    seed_dqn(config.seed)
    seed_generator = seed_training(config.seed, env.action_space)
    steps = 0

    for _ in tqdm(range(config.n_training_episodes)):
        seed = next_episode_seed(seed_generator)
        state, info = env.reset(seed=seed)
        raw_state = info.get("raw_state", state)
        if isinstance(agent, DQNRM):
            agent.reset_rm()

        for _ in range(config.max_steps):
            epsilon = compute_epsilon(
                config.min_epsilon,
                config.max_epsilon,
                steps,
                1 / config.dqn_epsilon_decay_steps,
            )
            action = agent.epsilon_greedy_policy(state, epsilon, env.action_space.sample)
            new_state, env_reward, terminated, truncated, info = env.step(action)
            new_raw_state = info.get("raw_state", new_state)

            if isinstance(agent, DQNRM):
                rm_done = agent.update(
                    state,
                    action,
                    raw_state,
                    new_raw_state,
                    new_state,
                    terminated,
                    env,
                    get_propositions,
                    use_crm=config.use_crm,
                )
            else:
                agent.update(state, action, env_reward, new_state, terminated)
                rm_done = False

            steps += 1
            if terminated or truncated or rm_done:
                break
            state = new_state
            raw_state = new_raw_state

    return agent
