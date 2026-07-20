from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.envs import create_environment
from src.models import QTableHRM, evaluate_agent, train_qtable_hrm
from src.utils import record_video


def train_hrm_agent(config: Configuration):
    """Train tabular hierarchical RL with a Reward Machine."""
    # ==================================================================
    #                       ENVIRONMENT & Q TABLES
    # ==================================================================
    print_separator("Environment", sep_type="LONG")
    env, get_propositions = create_environment(
        config.gym_id,
        multitaxi_grid_size=config.multitaxi_grid_size,
        multitaxi_num_passengers=config.multitaxi_num_passengers,
        multitaxi_observation_mode=config.multitaxi_observation_mode,
    )
    print(f" - Observation space: {env.observation_space}")
    print(f" - There are {env.action_space.n} possible actions")

    print_separator("HRM Q-Tables", sep_type="LONG")
    qtable = QTableHRM(config, env, config.rm_file)
    qtable.print_size()

    # ==================================================================
    #                               TRAINING
    # ==================================================================
    print_separator("TRAINING", sep_type="SUPER")
    qtable = train_qtable_hrm(config, qtable, get_propositions, env)

    # ==================================================================
    #                               TESTING
    # ==================================================================
    print_separator("TESTING", sep_type="SUPER")
    mean_reward, std_reward = evaluate_agent(config, qtable, get_propositions, env)
    print(f" - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    print_separator("VIDEO RECORDING", sep_type="LONG")
    record_video(
        config,
        qtable,
        env,
        get_propositions,
        video_name=f"{config.rm_file}_hrm_video.gif",
    )
    env.close()
