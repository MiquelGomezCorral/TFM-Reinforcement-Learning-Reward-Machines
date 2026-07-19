from maikol_utils.print_utils import print_separator

from src.models import QTableRM, train_qtable_crm, evaluate_agent
from src.utils import record_video
from src.envs import create_environment
from src.config import Configuration


def train_qt(CONFIG: Configuration):
    """Train QTable with Reward Machines."""
    # ==================================================================
    #                       ENVIRONMENT & Q TABLE
    # ==================================================================
    print_separator("Environment", sep_type="LONG")
    env, get_propositions = create_environment(CONFIG.gym_id)

    print(" - There are ", env.observation_space.n, " possible states")
    print(" - There are ", env.action_space.n, " possible actions")

    print_separator("Q-Table", sep_type="LONG")
    qt = QTableRM(
        CONFIG, env, rm_file=CONFIG.rm_file if CONFIG.use_rm else None,
        dynamic=CONFIG.dynamic_qtable,
    )
    qt.print_size()

    # ==================================================================
    #                               TRAINING
    # ==================================================================
    print_separator("TRAINING", sep_type="SUPER")
    qt = train_qtable_crm(CONFIG, qt, get_propositions, env)

    # ==================================================================
    #                               TESTING
    # ==================================================================
    print_separator("TESTING", sep_type="SUPER")
    mean_reward, std_reward = evaluate_agent(CONFIG, qt, get_propositions, env)
    print(f" - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    print_separator("VIDEO RECORDING", sep_type="LONG")
    record_video(CONFIG, qt, env, get_propositions, video_name=f"{CONFIG.rm_file}_qtable_video.gif")
    env.close()
