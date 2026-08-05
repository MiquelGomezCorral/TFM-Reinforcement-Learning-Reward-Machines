
## CRM DQN

- `app/src/models/DQNRM.py` currently prunes counterfactual replay using `_valid_crm_states` and `continue`.
- This is not the CRM algorithm described in `docs/RMs-Paper.pdf`, Section 4.2 / Algorithm 2: every real environment transition should generate one counterfactual experience for every non-final RM state.
- Remove `_valid_crm_states`, `reachable_states`, and the pruning condition. Keep `simulate_step` for each `u` and store every resulting transition in replay.
- Add a regression test asserting that each DQN update adds one replay transition per RM state on every step.

# Add hrm para DQM