from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.envs import create_environment
from src.models import QTableHRM, evaluate_agent, train_hrm
from src.utils import record_video


def train_hrm_agent(CONFIG: Configuration, progress_callback=None):
    """Train tabular hierarchical RL with a Reward Machine."""
    # ==================================================================
    #                       ENVIRONMENT & Q TABLES
    # ==================================================================
    print_separator("Environment", sep_type="LONG")
    env, get_propositions = create_environment(
        CONFIG
    )
    print(f" - Observation space: {env.observation_space}")
    print(f" - There are {env.action_space.n} possible actions")

    print_separator("HRM Q-Tables", sep_type="LONG")
    qtable = QTableHRM(CONFIG, env, CONFIG.rm_file)
    qtable.print_size()

    # ==================================================================
    #                               TRAINING
    # ==================================================================
    print_separator("TRAINING", sep_type="SUPER")
    qtable = train_hrm(CONFIG, qtable, get_propositions, env, progress_callback)

    # ==================================================================
    #                               TESTING
    # ==================================================================
    print_separator("TESTING", sep_type="SUPER")
    mean_reward, std_reward = evaluate_agent(CONFIG, qtable, get_propositions, env)
    print(f" - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    print_separator("VIDEO RECORDING", sep_type="LONG")
    record_video(
        CONFIG,
        qtable,
        env,
        get_propositions,
        video_name=f"{CONFIG.multitaxi_grid_size}x{CONFIG.multitaxi_grid_size}_{CONFIG.exp_name}_seed{CONFIG.seed}_video.gif",
    )
    env.close()
    return mean_reward, std_reward
