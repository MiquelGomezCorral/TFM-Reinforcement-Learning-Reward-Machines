import argparse

from src.config import Configuration
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

from scripts import train_dqn_agent, train_dqn_hrm_agent, train_qt, train_qt_hrm_agent


def cmd_train_qt(args: argparse.Namespace):
    """Call train_qt with the given args."""
    CONFIG = args_to_dataclass(args, Configuration)
    if args.seed is not None:
        CONFIG.set_seed(args.seed)
    print_separator("START TRAIN QTABLE", sep_type="START")
    train_qt(CONFIG)
    print_separator("END TRAIN QTABLE", sep_type="START")


def cmd_train_dqn(args: argparse.Namespace):
    config = args_to_dataclass(args, Configuration)
    if args.seed is not None:
        config.set_seed(args.seed)
    print_separator("START TRAIN DQN", sep_type="START")
    train_dqn_agent(config)
    print_separator("END TRAIN DQN", sep_type="START")


def cmd_train_qt_hrm(args: argparse.Namespace):
    config = args_to_dataclass(args, Configuration)
    if args.seed is not None:
        config.set_seed(args.seed)
    print_separator("START TRAIN QT HRM", sep_type="START")
    train_qt_hrm_agent(config)
    print_separator("END TRAIN QT HRM", sep_type="START")


def cmd_train_dqn_hrm(args: argparse.Namespace):
    config = args_to_dataclass(args, Configuration)
    if args.seed is not None:
        config.set_seed(args.seed)
    print_separator("START TRAIN DQN HRM", sep_type="START")
    train_dqn_hrm_agent(config)
    print_separator("END TRAIN DQN HRM", sep_type="START")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument(
        "--config", dest="yaml_config_path", metavar="CONFIG", default="default.yaml",
        help="Configuration file under configs/",
    )
    parser.add_argument("--seed", type=int, help="Override the configuration seed")

    subparsers = parser.add_subparsers(dest="function", required=True)

    p_train_qt = subparsers.add_parser("train-qt", help="Train QTable with optional Reward Machines")
    p_train_qt.set_defaults(func=cmd_train_qt)
    p_train_dqn = subparsers.add_parser("train-dqn", help="Train DQN with optional Reward Machines")
    p_train_dqn.set_defaults(func=cmd_train_dqn)
    p_train_qt_hrm = subparsers.add_parser("train-qt-hrm", help="Train tabular HRM")
    p_train_qt_hrm.set_defaults(func=cmd_train_qt_hrm)
    p_train_dqn_hrm = subparsers.add_parser("train-dqn-hrm", help="Train Deep HRM")
    p_train_dqn_hrm.set_defaults(func=cmd_train_dqn_hrm)
    args = parser.parse_args()
    args.func(args)
