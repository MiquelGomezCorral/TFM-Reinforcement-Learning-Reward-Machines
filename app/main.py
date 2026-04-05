"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from src.config import Configuration
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

from scripts import train_qt

def cmd_train_qt(args: argparse.Namespace):
    """Call train_qt with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START TRAIN QTABLE", sep_type="START")
    train_qt(CONFIG)
    print_separator("END TRAIN QTABLE", sep_type="START")

def cmd_test(args):
    """Call test functions."""
    ...

# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Name of the config file at configs/ (default: config.yaml)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       Train QTable RM
    # ======================================================================================
    p_train = subparsers.add_parser("train-qrm", help="Train QTable with Reward Machines")
    p_train.add_argument(
        "-d", "--dataset_name", type=str, default="Nuelas", help="Name of raw data folder"
    )
    p_train.add_argument("-m", "--max_files", type=int, default=None, help="Max files to load")
    p_train.add_argument(
        "-l", "--use_llm", action="store_false", default=True, help="Disable LLM extraction"
    )
    p_train.set_defaults(func=cmd_train_qt)

    # ======================================================================================
    #                                       test
    # ======================================================================================
    p_test = subparsers.add_parser("test", help="Test script with any code")
    p_test.set_defaults(func=cmd_test)

    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)
