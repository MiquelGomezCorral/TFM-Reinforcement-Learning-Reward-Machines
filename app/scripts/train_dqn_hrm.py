from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.envs import create_environment
from src.models import DQNHRM, evaluate_agent, train_dqn_hrm
from src.utils import record_video, seed_dqn


def train_dhrm_agent(CONFIG: Configuration, progress_callback=None):
    seed_dqn(CONFIG.seed)
    env, get_propositions = create_environment(
        CONFIG,
        one_hot_discrete=True,
    )

    print_separator("Deep HRM", sep_type="LONG")
    agent = DQNHRM(CONFIG, env, CONFIG.rm_file)
    agent.print_size()
    agent = train_dqn_hrm(CONFIG, agent, get_propositions, env, progress_callback)
    mean_reward, std_reward = evaluate_agent(CONFIG, agent, get_propositions, env)
    print(f" - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    record_video(
        CONFIG,
        agent,
        env,
        get_propositions,
        video_name=f"{CONFIG.multitaxi_grid_size}x{CONFIG.multitaxi_grid_size}_{CONFIG.exp_name}_seed{CONFIG.seed}_video.gif",
    )
    env.close()
