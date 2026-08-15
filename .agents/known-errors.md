# Known Errors

## Errors

### Plain DQN Learns Pickup But Never Completes One Passenger

- Symptom: greedy evaluation remains 0/100 with mean reward near -94, usually indicating a learned `+5` pickup followed by a timeout.
- Proven non-causes: replay memory contains `+20` terminal drop-offs; normalizing observations and extending warm-up/epsilon decay alone do not make 48- or 64-unit networks solve seed 43.
- Current resolution: normalized relative observations with two 128-unit hidden layers and 30,000 episodes reached stable 100/100 shaped-reward evaluation on development seeds 42, 43, and 44.
- Trap: best-checkpoint restoration can hide later policy regression. Judge the final policy and inspect intermediate evaluations only for diagnosis.

### Relative Observation Omits Taxi Position

- Symptom: different raw states can share the same relative passenger/destination input and have different movement transitions at a boundary.
- Evidence: exhaustive 5x5 enumeration found 172 aliased two-passenger states. Adding normalized taxi row and column removes every alias without providing a grid map or factored state.
- Resolution: do not add coordinates to the compact baseline without a separate capacity study. The coordinate-enhanced one-passenger seed-43 run reached only 76/100 at 30,000 episodes; this baseline keeps relative passenger/destination features only.
