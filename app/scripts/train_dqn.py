from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.envs import create_environment
from src.models import DQN, DQNRM, evaluate_agent, train_dqn
from src.utils import record_video, seed_dqn


def train_dqn_agent(CONFIG: Configuration, progress_callback=None):
    if CONFIG.use_crm and not CONFIG.use_rm:
        raise ValueError("use_crm requires use_rm=True")

    seed_dqn(CONFIG.seed)
    env, get_propositions = create_environment(
        CONFIG,
        one_hot_discrete=True,
    )
    if CONFIG.use_rm:
        agent = DQNRM(CONFIG, env, CONFIG.rm_file)
    else:
        agent = DQN(
            input_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
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
    agent = train_dqn(CONFIG, agent, get_propositions, env, progress_callback)
    mean_reward, std_reward = evaluate_agent(CONFIG, agent, get_propositions, env)
    print(f" - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    grid = f"{CONFIG.multitaxi_grid_size}x{CONFIG.multitaxi_grid_size}"
    record_video(CONFIG, agent, env, get_propositions, video_name=f"{grid}_{CONFIG.exp_name}_seed{CONFIG.seed}_video.gif")
    env.close()
    return mean_reward, std_reward
