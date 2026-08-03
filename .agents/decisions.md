# Decisions

Record closed decisions here. Do not reopen them unless the user explicitly asks.

## Active Decisions

### Training Pipeline

- Notebooks orchestrate the command implementations in `app/scripts/`; they do not duplicate model construction, training, evaluation, or video logic.
- DQN acceptance uses environment reward from full configured `app/main.py` runs.
- Keep DQN variants structurally similar to the Q-table pipeline. Add DQN-specific behavior only when real CLI measurements show it is needed.
