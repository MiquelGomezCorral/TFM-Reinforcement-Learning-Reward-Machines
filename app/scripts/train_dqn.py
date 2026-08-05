from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.envs import create_environment, create_vector_environment
from src.models import DQN, DQNRM, evaluate_agent, train_dqn
from src.utils import record_video, seed_dqn


def train_dqn_agent(CONFIG: Configuration, progress_callback=None):
    if CONFIG.use_crm and not CONFIG.use_rm:
        raise ValueError("use_crm requires use_rm=True")
    if CONFIG.dqn_num_envs > 1 and CONFIG.use_rm:
        raise ValueError("Vectorized training currently supports plain DQN only")

    seed_dqn(CONFIG.seed)
    evaluation_env, get_propositions = create_environment(
        CONFIG,
        one_hot_discrete=True,
    )
    env = (
        create_vector_environment(CONFIG, CONFIG.dqn_num_envs, one_hot_discrete=True)
        if CONFIG.dqn_num_envs > 1 else evaluation_env
    )
    observation_space = getattr(env, "single_observation_space", env.observation_space)
    if CONFIG.use_rm:
        agent = DQNRM(CONFIG, env, CONFIG.rm_file)
    else:
        agent = DQN(
            input_size=observation_space.shape[0],
            action_size=env.single_action_space.n if CONFIG.dqn_num_envs > 1 else env.action_space.n,
            batch_size=CONFIG.dqn_batch_size,
            replay_capacity=CONFIG.dqn_replay_capacity,
            learning_rate=CONFIG.dqn_learning_rate,
            gamma=CONFIG.gamma,
            hidden_size=CONFIG.dqn_hidden_size,
            tau=CONFIG.dqn_tau,
            gradient_clip=CONFIG.dqn_gradient_clip,
        )

    print_separator("DQN", sep_type="LONG")
    agent.print_size()
    agent = train_dqn(
        CONFIG,
        agent,
        get_propositions,
        env,
        progress_callback,
        evaluation_env=evaluation_env,
    )
    mean_reward, std_reward = evaluate_agent(CONFIG, agent, get_propositions, evaluation_env)
    print(f" - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    grid = f"{CONFIG.multitaxi_grid_size}x{CONFIG.multitaxi_grid_size}"
    record_video(
        CONFIG,
        agent,
        evaluation_env,
        get_propositions,
        video_name=f"{grid}_{CONFIG.exp_name}_seed{CONFIG.seed}_video.gif",
    )
    if env is not evaluation_env:
        env.close()
    evaluation_env.close()
    return mean_reward, std_reward
