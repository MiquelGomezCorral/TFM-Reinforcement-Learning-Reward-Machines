import minigrid
import gymnasium as gym

from maikol_utils.print_utils import print_separator

from src.models import QTable, train_qtable, evaluate_agent
from src.utils import record_video
from src.envs import get_propositions_taxi, get_propositions_doorkey, MiniGridDiscreteWrapper
from src.config import Configuration


def train_qt(CONFIG: Configuration):
    """Train QTable with Reward Machines."""
    # ==================================================================
    #                       ENVIRONMENT & Q TABLE
    # ==================================================================
    print_separator("Environment", sep_type="LONG")
    env = gym.make(CONFIG.gym_id, render_mode="rgb_array")
    env = MiniGridDiscreteWrapper(env)

    state_space = env.observation_space.n
    print(" - There are ", state_space, " possible states")
    action_space = env.action_space.n
    print(" - There are ", action_space, " possible actions")

    print_separator("Q-Table", sep_type="LONG")
    qt = QTable(CONFIG, env, rm_file=CONFIG.rm_file if CONFIG.use_rm else None)
    qt.print_size()

    # ==================================================================
    #                               TRAINING
    # ==================================================================
    print_separator("TRAINING", sep_type="SUPER")
    qt = train_qtable(CONFIG, qt, get_propositions_doorkey, env)

    # ==================================================================
    #                               TESTING
    # ==================================================================
    print_separator("TESTING", sep_type="SUPER")
    mean_reward, std_reward = evaluate_agent(CONFIG, qt, get_propositions_doorkey, env)
    print(f" - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


    print_separator("VIDEO RECORDING", sep_type="LONG")
    record_video(CONFIG, qt, env, get_propositions_doorkey, video_name=f"{CONFIG.rm_file}_qtable_video.gif")