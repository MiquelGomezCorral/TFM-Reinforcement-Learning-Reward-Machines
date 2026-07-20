from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.envs import create_environment
from src.models import DQNHRM, evaluate_agent, train_dqn_hrm
from src.utils import record_video, seed_dqn


def train_dhrm_agent(config: Configuration):
    seed_dqn(config.seed)
    env, get_propositions = create_environment(
        config.gym_id,
        one_hot_discrete=True,
        multitaxi_grid_size=config.multitaxi_grid_size,
        multitaxi_num_passengers=config.multitaxi_num_passengers,
        multitaxi_observation_mode=config.multitaxi_observation_mode,
    )

    print_separator("Deep HRM", sep_type="LONG")
    agent = DQNHRM(config, env, config.rm_file)
    agent.print_size()
    agent = train_dqn_hrm(config, agent, get_propositions, env)
    mean_reward, std_reward = evaluate_agent(config, agent, get_propositions, env)
    print(f" - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    record_video(
        config,
        agent,
        env,
        get_propositions,
        video_name=f"{config.rm_file}_dhrm_video.gif",
    )
    env.close()
