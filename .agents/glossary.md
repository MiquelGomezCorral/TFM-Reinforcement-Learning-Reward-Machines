# Glossary

## Terms

- Reward Machine (RM): finite-state reward specification loaded from a file in `models/`; transitions match environment propositions and produce rewards.
- Q-table: action-value storage indexed by reward-machine state and environment state.
- Proposition: event label extracted from an environment state and supplied to a reward machine.
- CRM: configuration option named `use_crm`; its meaning is not documented in the repository.
