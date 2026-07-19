from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.envs import create_environment
from src.models import DQN, DQNRM, evaluate_agent, train_dqn
from src.utils import record_video, seed_dqn


def train_dqn_agent(config: Configuration):
    seed_dqn(config.seed)
    env, get_propositions = create_environment(
        config.gym_id,
        one_hot_discrete=True,
        multitaxi_grid_size=config.multitaxi_grid_size,
        multitaxi_num_passengers=config.multitaxi_num_passengers,
        multitaxi_observation_mode=config.multitaxi_observation_mode,
    )
    if config.use_rm:
        agent = DQNRM(config, env, config.rm_file)
    else:
        agent = DQN(
            input_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            batch_size=config.dqn_batch_size,
            replay_capacity=config.dqn_replay_capacity,
            learning_rate=config.dqn_learning_rate,
            gamma=config.gamma,
            hidden_size=config.dqn_hidden_size,
            tau=config.dqn_tau,
            gradient_clip=config.dqn_gradient_clip,
        )

    print_separator("DQN", sep_type="LONG")
    agent.print_size()
    agent = train_dqn(config, agent, get_propositions, env)
    mean_reward, std_reward = evaluate_agent(config, agent, get_propositions, env)
    print(f" - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    record_video(config, agent, env, get_propositions, video_name=f"{config.rm_file}_dqn_video.gif")
    env.close()
