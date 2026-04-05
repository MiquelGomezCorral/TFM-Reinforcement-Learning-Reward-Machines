import numpy as np
from .RewardMachine import RewardMachine

class QTable:
    def __init__(self, CONFIG, env, rm_file: str = None):
        self.CONFIG = CONFIG
        self.rm = RewardMachine(self.CONFIG, rm_file) if rm_file else None

        self.initialize_q_table(
            self.rm.get_num_states() if self.rm else 1,
            env.observation_space.n, 
            env.action_space.n
        )

    def initialize_q_table(self, num_u_states, state_space, action_space):
        """
        Is not a matrix, is an array and we can locate each game cell later with `current_row * ncols + current_col`
        """
        self.Qtable = np.zeros((num_u_states, state_space, action_space))

    def print_size(self):
        print(f" - Q-Table size: {self.Qtable.shape}")

    def get_rm_state(self):
        return self.rm.get_current_state() if self.rm else 0
    
    def step_rm(self, events):
        if self.rm:
            return self.rm.step(events)
        return 0, 0, False
    
    def reset_rm(self):
        if self.rm:
            self.rm.reset()

    def greedy_policy(self, state, u = None):
        """Take the action with the highest value given a state
        """
        if u is None:
            u = self.get_rm_state()

        action = np.argmax(self.Qtable[u][state][:])
        return action

    def epsilon_greedy_policy(self, state, epsilon, env):
        """Take the action with the highest value given a state with a probability of 1-epsilon, otherwise take a random action
        """
        random_num = np.random.random()

        if random_num > epsilon:
            action = self.greedy_policy(state)
        else:
            action = env.action_space.sample() 

        return action

    def _update_q_value(self, u, state, action, reward, target_u, new_state, done, gamma, learning_rate):
        """Helper to apply the Bellman equation to a specific RM state table."""
        old_q = self.Qtable[u][state][action]
        td_target = reward if done else reward + gamma * np.max(self.Qtable[target_u][new_state])
        self.Qtable[u][state][action] = old_q + learning_rate * (td_target - old_q)

    def update(self, state, action, env_reward, new_state, gamma, learning_rate, env, get_propositions, use_crm=False):
        """Update the Q-table using the Q-learning update rule"""
        current_u = self.get_rm_state()
        
        done = False
        target_u = current_u
        reward = env_reward

        if self.rm:
            events = get_propositions(env, new_state)
            
            # CRM logic toggle 
            if use_crm:
                for u in self.rm.states.keys():
                    t_u, r_u, d_u = self.rm.simulate_step(u, events)
                    self._update_q_value(
                        u, state, action, r_u, 
                        t_u, new_state, d_u, 
                        gamma, learning_rate
                    )

            # Advance actual RM state
            target_u, reward, done = self.step_rm(events)
        
        # Standard update for the current state
        self._update_q_value(
            current_u, state, action, reward, 
            target_u, new_state, 
            done, gamma, learning_rate
        )

        return done