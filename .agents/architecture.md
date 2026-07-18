# Architecture

## Stack

Python package built with setuptools. The source uses Gymnasium, MiniGrid, NumPy, PyYAML, `maikol-utils`, and `dotenv`.

## Layout

- `app/main.py`: CLI entry point.
- `app/scripts/`: command implementations.
- `app/src/config/`: configuration dataclass and YAML loading.
- `app/src/envs/`: Taxi and DoorKey environments, wrappers, and proposition extraction.
- `app/src/models/`: Q-table, reward machine, training, and evaluation logic.
- `configs/`: YAML experiment configurations.
- `models/`: reward-machine definitions and serialized models.
- `data/`, `logs/`, `videos/`: runtime output directories.
- `notebooks/`: experiments and analysis.
- `docs/`: project documentation and reference material.

## Boundaries

`Configuration` supplies paths and training parameters. Environment modules expose observations and propositions. `QTable` optionally owns a `RewardMachine`; training, evaluation, and video recording use the configured environment and Q-table.

## Entry Points

`app/main.py` defines the `train-qrm` and `test` CLI subcommands. `train-qrm` calls `scripts.train_qt`.
