import random

import numpy as np

from .DQN import DQN
from .HRM import HRM


class DQNHRM(HRM):
    def __init__(self, config, env, rm_file):
        if len(env.observation_space.shape) != 1:
            raise ValueError("DQNHRM requires one-dimensional vector observations")

        super().__init__(config, rm_file)
        self._rm_indices = {u: index for index, u in enumerate(self.rm_states)}

        observation_size = env.observation_space.shape[0]
        self.high_level = self._new_dqn(
            config,
            observation_size + len(self.rm_states),
            len(self.target_states),
        )
        self.actor = self._new_dqn(
            config,
            observation_size + len(self.rm_states) + len(self.target_states),
            env.action_space.n,
        )

    @staticmethod
    def _new_dqn(config, input_size, action_size):
        return DQN(
            input_size=input_size,
            action_size=action_size,
            batch_size=config.dqn_batch_size,
            replay_capacity=config.dqn_replay_capacity,
            learning_rate=config.dqn_learning_rate,
            gamma=config.gamma,
            hidden_size=config.dqn_hidden_size,
            tau=config.dqn_tau,
            gradient_clip=config.dqn_gradient_clip,
        )

    def _rm_state(self, u):
        state = np.zeros(len(self.rm_states), dtype=np.float32)
        state[self._rm_indices[u]] = 1.0
        return state

    def _target_state(self, target_u):
        state = np.zeros(len(self.target_states), dtype=np.float32)
        state[self.target_action(target_u)] = 1.0
        return state

    @staticmethod
    def _environment_state(state):
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def high_state(self, state, u):
        return np.concatenate((self._environment_state(state), self._rm_state(u)))

    def actor_state(self, state, u, target_u):
        return np.concatenate((
            self._environment_state(state),
            self._rm_state(u),
            self._target_state(target_u),
        ))

    def max_high_value(self, state, u):
        actions = self.option_actions(u)
        values = self.high_level.q_values(self.high_state(state, u), target=True)
        return float(np.max(values[list(actions)]))

    def select_option(self, state, epsilon, u=None):
        u = self.get_rm_state() if u is None else u
        targets = self.options[u]
        if random.random() < epsilon:
            target_u = random.choice(targets)
        else:
            actions = self.option_actions(u)
            values = self.high_level.q_values(self.high_state(state, u))[list(actions)]
            target_u = targets[int(np.argmax(values))]
        self.active_option = target_u
        return target_u

    def print_size(self):
        print(" - High-level", end="")
        self.high_level.print_size()
        print(" - Actor", end="")
        self.actor.print_size()

    def greedy_policy(self, state, u=None):
        u = self.get_rm_state() if u is None else u
        if self.active_option not in self.options[u]:
            self.select_option(state, 0, u)
        return self.actor.greedy_policy(self.actor_state(state, u, self.active_option))

    def epsilon_greedy_policy(self, state, epsilon, sample_action=None):
        u = self.get_rm_state()
        if self.active_option not in self.options[u]:
            self.select_option(state, epsilon, u)
        return self.actor.epsilon_greedy_policy(
            self.actor_state(state, u, self.active_option),
            epsilon,
            sample_action,
        )
