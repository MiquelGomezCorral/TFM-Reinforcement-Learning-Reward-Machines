import numpy as np

from .DQN import DQN
from .RewardMachine import RewardMachine


class DQNRM:
    def __init__(self, config, env, rm_file):
        if len(env.observation_space.shape) != 1:
            raise ValueError("DQNRM requires one-dimensional vector observations")

        self.rm = RewardMachine(config, rm_file)
        self._rm_states = self._states()
        self._rm_indices = {u: index for index, u in enumerate(self._rm_states)}
        crm_multiplier = len(self._rm_states) if config.use_crm else 1
        self.dqn = DQN(
            input_size=env.observation_space.shape[0] + len(self._rm_states),
            action_size=env.action_space.n,
            batch_size=config.dqn_batch_size * crm_multiplier,
            replay_capacity=config.dqn_replay_capacity * crm_multiplier,
            learning_rate=config.dqn_learning_rate,
            gamma=config.gamma,
            hidden_size=config.dqn_hidden_size,
            tau=config.dqn_tau,
            gradient_clip=config.dqn_gradient_clip,
            rewarding_fraction=0.25,
        )

    def _states(self):
        return sorted({self.rm.initial_state, *self.rm.states})

    def _state(self, state, u):
        rm_state = np.zeros(len(self._rm_states), dtype=np.float32)
        rm_state[self._rm_indices[u]] = 1.0
        return np.concatenate((np.asarray(state, dtype=np.float32).reshape(-1), rm_state))

    def print_size(self):
        self.dqn.print_size()

    def get_rm_state(self):
        return self.rm.get_current_state()

    def step_rm(self, events):
        return self.rm.step(events)

    def reset_rm(self):
        self.rm.reset()

    def greedy_policy(self, state, u=None):
        return self.dqn.greedy_policy(self._state(state, self.get_rm_state() if u is None else u))

    def epsilon_greedy_policy(self, state, epsilon, sample_action=None):
        return self.dqn.epsilon_greedy_policy(
            self._state(state, self.get_rm_state()), epsilon, sample_action
        )

    def update(
        self,
        state,
        action,
        env_reward,
        raw_state,
        new_raw_state,
        new_state,
        terminated,
        env,
        get_propositions,
        use_crm=False,
        optimize=True,
    ):
        events = get_propositions(env, raw_state, action, new_raw_state)
        current_u = self.get_rm_state()
        target_u, reward, rm_done = self.step_rm(events)

        if use_crm:
            for u in self._rm_states:
                next_u, counterfactual_reward, counterfactual_done = self.rm.simulate_step(u, events)
                terminal = terminated or counterfactual_done
                self.dqn.remember(
                    self._state(state, u),
                    action,
                    min(env_reward, counterfactual_reward),
                    None if terminal else self._state(new_state, next_u),
                    terminal,
                )
        else:
            terminal = terminated or rm_done
            self.dqn.remember(
                self._state(state, current_u),
                action,
                reward,
                None if terminal else self._state(new_state, target_u),
                terminal,
            )

        if optimize:
            self.dqn.optimize()
        return rm_done
