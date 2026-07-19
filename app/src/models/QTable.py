import numpy as np
from .RewardMachine import RewardMachine

class QTable:
    def __init__(self, state_space, action_space, dynamic: bool = True):
        self.action_space = action_space
        self.dynamic = dynamic
        self._table = {} if dynamic else np.zeros((state_space, action_space))

    def _state_key(self, state):
        return tuple(state) if isinstance(state, np.ndarray) else state

    def _values(self, state):
        if not self.dynamic:
            return self._table[state]

        key = self._state_key(state)
        if key not in self._table:
            self._table[key] = np.zeros(self.action_space)
        return self._table[key]

    def print_size(self):
        if self.dynamic:
            print(f" - Q-Table entries: {len(self._table)} states")
        else:
            print(f" - Q-Table size: {self._table.shape}")

    def greedy_policy(self, state):
        return int(np.argmax(self._values(state)))

    def epsilon_greedy_policy(self, state, epsilon, sample_action=None):
        if np.random.random() > epsilon:
            return self.greedy_policy(state)
        return sample_action() if sample_action else int(np.random.randint(self.action_space))

    def update(self, state, action, reward, new_state, done, gamma, learning_rate, next_q_table=None):
        q_values = self._values(state)
        next_q_table = self if next_q_table is None else next_q_table
        future = 0 if done else gamma * np.max(next_q_table._values(new_state))
        q_values[action] += learning_rate * (reward + future - q_values[action])


class QTableRM:
    def __init__(self, CONFIG, env, rm_file: str = None, dynamic: bool = True):
        print(f"The QTable is {'dynamic' if dynamic else 'static'}")
        self.rm = RewardMachine(CONFIG, rm_file) if rm_file else None
        self.state_space = None if dynamic else env.observation_space.n
        self.action_space = env.action_space.n
        rm_states = self.rm.states if self.rm else [0]
        self._q_tables = {u: self._new_q_table() for u in rm_states}

    def _new_q_table(self):
        return QTable(self.state_space, self.action_space, dynamic=self.state_space is None)

    def _q_table(self, u):
        if u not in self._q_tables:
            self._q_tables[u] = self._new_q_table()
        return self._q_tables[u]

    def print_size(self):
        for u, q_table in self._q_tables.items():
            print(f" - RM state {u}", end="")
            q_table.print_size()

    def get_rm_state(self):
        return self.rm.get_current_state() if self.rm else 0

    def step_rm(self, events):
        if self.rm:
            return self.rm.step(events)
        return 0, 0, False

    def reset_rm(self):
        if self.rm:
            self.rm.reset()

    def greedy_policy(self, state, u=None):
        if u is None:
            u = self.get_rm_state()
        return self._q_table(u).greedy_policy(state)

    def epsilon_greedy_policy(self, state, epsilon, env):
        return self._q_table(self.get_rm_state()).epsilon_greedy_policy(
            state, epsilon, env.action_space.sample
        )

    def update(
        self, state, action, env_reward, raw_state, new_state, new_state_parse,
        gamma, learning_rate, env, get_propositions, terminated=False, use_crm=False,
    ):
        current_u = self.get_rm_state()
        rm_done = False

        if self.rm:
            events = get_propositions(env, raw_state, action, new_state)

            target_u, reward, rm_done = self.step_rm(events)
            if use_crm:
                for u in self.rm.states:
                    t_u, r_u, d_u = self.rm.simulate_step(u, events)
                    done = terminated or d_u
                    self._q_table(u).update(
                        state, action, r_u, new_state_parse, done, gamma, learning_rate,
                        None if done else self._q_table(t_u),
                    )
            else:
                done = terminated or rm_done
                self._q_table(current_u).update(
                    state, action, reward, new_state_parse, done, gamma, learning_rate,
                    None if done else self._q_table(target_u),
                )
        else:
            self._q_table(current_u).update(
                state, action, env_reward, new_state_parse, terminated, gamma, learning_rate
            )

        return rm_done
