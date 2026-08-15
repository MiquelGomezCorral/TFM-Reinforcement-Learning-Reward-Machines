# Decisions

Record closed decisions here. Do not reopen them unless the user explicitly asks.

## Active Decisions

### Training Pipeline

- Notebooks orchestrate the command implementations in `app/scripts/`; they do not duplicate model construction, training, evaluation, or video logic.
- DQN acceptance uses environment reward from full configured `app/main.py` runs.
- Keep DQN variants structurally similar to the Q-table pipeline. Add DQN-specific behavior only when real CLI measurements show it is needed.

### Plain DQN Baseline

- Establish plain DQN before testing Reward Machines: one passenger first, then two passengers with the same pipeline.
- The initial 48- and 64-unit candidates failed one-passenger seed 43 despite terminal replay coverage. The current one-passenger candidate uses two 128-unit hidden layers and 30,000 episodes; it reached stable 100/100 evaluation on seeds 42, 43, and 44 during development.
- The two-passenger dense-shaping candidate uses two 256-unit hidden layers and up to 100,000 episodes in `configs/dqn_2p.yaml`. Do not override its passenger count, width, or duration from the notebook.
- Change one experimental variable at a time and use fresh, fully specified result records.
- DQN observations contain normalized relative passenger/destination distances and one-hot passenger status. Do not expose a grid map or factored environment state.
- The default DQN baseline additionally enables dense distance-progress rewards on top of pickup/drop-off event rewards. This is tested before sparse rewards; the Q-table and other configurations keep it disabled unless explicitly requested.
- MultiTaxi intentionally has unlimited capacity and one action affects all eligible co-located passengers.
- Use shaped pickup/drop-off rewards until both passenger counts learn reliably, then change only the reward to the sparse completion reward.
- Exploration schedules are measured in collected transitions and must state warm-up and decay behavior unambiguously.
- Development may use intermediate success rates diagnostically, but an accepted baseline must solve 100/100 fixed held-out evaluation episodes and produce representative successful videos. Exhaustive valid initial-state evaluation is desirable after that criterion passes.
- Do not select a best checkpoint during baseline development. Evaluate the final policy, use multiple training seeds, and allow up to 200,000 episodes for the two-passenger task.
