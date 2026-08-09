from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.envs import create_environment
from src.models import DQNHRM, evaluate_agent, train_dqn_hrm
from src.utils import record_video


def train_dqn_hrm_agent(CONFIG: Configuration, progress_callback=None):
    env, get_propositions = create_environment(
        CONFIG,
        one_hot_discrete=True,
    )

    print_separator("Deep HRM", sep_type="LONG")
    agent = DQNHRM(CONFIG, env, CONFIG.rm_file)
    agent.print_size()
    agent = train_dqn_hrm(CONFIG, agent, get_propositions, env, progress_callback)
    metrics = evaluate_agent(
        CONFIG,
        agent,
        get_propositions,
        env,
        return_metrics=True,
    )
    print(f" - Mean_reward={metrics['mean_reward']:.2f} +/- {metrics['reward_std']:.2f}")
    record_video(
        CONFIG,
        agent,
        env,
        get_propositions,
        video_name=f"{CONFIG.multitaxi_grid_size}x{CONFIG.multitaxi_grid_size}_{CONFIG.exp_name}_seed{CONFIG.seed}_video.gif",
    )
    env.close()
    return metrics
