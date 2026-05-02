import numpy as np
from .RewardMachine import RewardMachine

class _StaticStorage:
    def __init__(self, num_u_states, state_space, action_space):
        self._table = np.zeros((num_u_states, state_space, action_space))

    def get(self, u, state):
        # print(f'{u = }')
        # print(f'{state = }')
        return self._table[u, state]       
    
    def set(self, u, state, values):
        self._table[u, state] = values

    def print_size(self):
        print(f" - Q-Table size: {self._table.shape}")


class _DynamicStorage:
    def __init__(self, action_space):
        self._table = {}
        self._action_space = action_space

    def _ensure(self, u, state):
        key = (u, state) if not isinstance(state, np.ndarray) else (u, tuple(state))
        if key not in self._table:
            self._table[key] = np.zeros((self._action_space,))  # explicit shape tuple

        # print(f"Ensured key: {key}, total entries: {len(self._table)}")
        return key

    def get(self, u, state):
        key = self._ensure(u, state)
        return self._table[key]

    def set(self, u, state, values):
        key = self._ensure(u, state)
        self._table[key] = values

    def print_size(self):
        print(f" - Q-Table entries: {len(self._table)} (u, state) pairs")


class QTable:
    def __init__(self, CONFIG, env, rm_file: str = None, dynamic: bool = True):
        self.CONFIG = CONFIG
        print(f'The QTable is {'dynamic' if dynamic else 'static'}')
        self.rm = RewardMachine(self.CONFIG, rm_file) if rm_file else None

        num_u = self.rm.get_num_states() if self.rm else 1
        action_space = env.action_space.n

        if dynamic:
            self._storage = _DynamicStorage(action_space)
        else:
            self._storage = _StaticStorage(num_u, env.observation_space.n, action_space)

    def print_size(self):
        self._storage.print_size()

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
        return int(np.argmax(self._storage.get(u, state)))

    def epsilon_greedy_policy(self, state, epsilon, env):
        if np.random.random() > epsilon:
            return self.greedy_policy(state)
        return env.action_space.sample()

    def _update_q_value(self, u, state, action, reward, target_u, new_state, done, gamma, learning_rate):
        q_values = self._storage.get(u, state).copy()
        future = 0 if done else gamma * np.max(self._storage.get(target_u, new_state))
        # print(q_values.shape)
        # print(q_values)
        q_values[action] = q_values[action] + learning_rate * (reward + future - q_values[action])
        self._storage.set(u, state, q_values)

    def update(self, state, action, env_reward, new_state, new_state_parse, gamma, learning_rate, env, get_propositions, use_crm=False, skip_first_rm_state=False):
        current_u = self.get_rm_state()
        done = False
        target_u = current_u
        reward = env_reward

        if self.rm:
            events = get_propositions(env, new_state)

            target_u, reward, done = self.step_rm(events)
            if use_crm:
                for u in self.rm.states.keys():
                    if skip_first_rm_state and u == 0:
                        continue  # Skip the first RM state
                    t_u, r_u, d_u = self.rm.simulate_step(u, events)
                    self._update_q_value(u, state, action, r_u, t_u, new_state_parse, d_u, gamma, learning_rate)
            else:
                self._update_q_value(current_u, state, action, reward, target_u, new_state_parse, done, gamma, learning_rate)
            
            # print(f"RM transition: {current_u} --{events}--> {target_u}, reward: {reward}, done: {done}")
        else:
            self._update_q_value(current_u, state, action, reward, target_u, new_state_parse, done, gamma, learning_rate)

        return done