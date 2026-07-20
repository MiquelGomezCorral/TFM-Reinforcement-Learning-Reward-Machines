import numpy as np

from .QTable import QTable
from .RewardMachine import RewardMachine


class QTableHRM:
    def __init__(self, config, env, rm_file):
        self.rm = RewardMachine(config, rm_file)
        self.rm_states = tuple(sorted(self.rm.states))
        self.target_states = tuple(sorted({*self.rm_states, self.rm.final_state}))
        self._target_actions = {
            target: action for action, target in enumerate(self.target_states)
        }
        self.options = {
            u: tuple(sorted({u, *(target for target, _, _ in transitions)}))
            for u, transitions in self.rm.states.items()
        }
        self.high_level = QTable(
            None,
            len(self.target_states),
            initial_value=config.hrm_q_init,
        )
        self.actor = QTable(
            None,
            env.action_space.n,
            initial_value=config.hrm_q_init,
        )
        self.active_option = None

    @staticmethod
    def _environment_state(state):
        if isinstance(state, np.ndarray):
            return tuple(state.reshape(-1).tolist())
        if isinstance(state, list):
            return tuple(state)
        return state

    def high_state(self, state, u):
        return self._environment_state(state), u

    def actor_state(self, state, u, target_u):
        return self._environment_state(state), u, target_u

    def target_action(self, target_u):
        return self._target_actions[target_u]

    def option_actions(self, u):
        return tuple(self.target_action(target_u) for target_u in self.options[u])

    def max_high_value(self, state, u):
        actions = self.option_actions(u)
        return float(np.max(self.high_level.values(self.high_state(state, u))[list(actions)]))

    def select_option(self, state, epsilon, u=None):
        u = self.get_rm_state() if u is None else u
        targets = self.options[u]
        if np.random.random() < epsilon:
            target_u = targets[int(np.random.randint(len(targets)))]
        else:
            actions = self.option_actions(u)
            values = self.high_level.values(self.high_state(state, u))[list(actions)]
            target_u = targets[int(np.argmax(values))]
        self.active_option = target_u
        return target_u

    def print_size(self):
        print(" - High-level", end="")
        self.high_level.print_size()
        print(" - Actor", end="")
        self.actor.print_size()

    def get_rm_state(self):
        return self.rm.get_current_state()

    def reset_rm(self):
        self.rm.reset()
        self.active_option = None

    def step_rm(self, events):
        current_u = self.get_rm_state()
        next_u, reward, done = self.rm.step(events)
        if next_u != current_u or done:
            self.active_option = None
        return next_u, reward, done

    def greedy_policy(self, state, u=None):
        u = self.get_rm_state() if u is None else u
        if self.active_option not in self.options[u]:
            self.select_option(state, 0, u)
        return self.actor.greedy_policy(self.actor_state(state, u, self.active_option))

    def epsilon_greedy_policy(self, state, epsilon, env):
        u = self.get_rm_state()
        if self.active_option not in self.options[u]:
            self.select_option(state, epsilon, u)
        return self.actor.epsilon_greedy_policy(
            self.actor_state(state, u, self.active_option),
            epsilon,
            env.action_space.sample,
        )
