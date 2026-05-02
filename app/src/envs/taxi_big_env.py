import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiTaxiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size=5, num_passengers=2):
        self.grid_size = grid_size
        self.num_passengers = num_passengers

        if grid_size == 5:
            self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        elif grid_size == 10:
            self.locs = [(0, 0), (0, 9), (9, 0), (9, 8)]
        else:
            raise ValueError("grid_size must be 5 or 10")

        # State bounds: [taxi_r, taxi_c] + [p_loc, p_dest] * num_passengers
        # p_loc: 0-3 (locations), 4 (in taxi) | p_dest: 0-3 (locations)
        self.state_bounds = [grid_size, grid_size] + [5, 4] * num_passengers
        
        self.observation_space = spaces.Discrete(np.prod(self.state_bounds))
        self.action_space = spaces.Discrete(6) # 0:S, 1:N, 2:E, 3:W, 4:Pickup, 5:Dropoff
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        taxi_r = self.np_random.integers(self.grid_size)
        taxi_c = self.np_random.integers(self.grid_size)

        state_tuple = [taxi_r, taxi_c]
        for _ in range(self.num_passengers):
            p_loc = self.np_random.integers(4)
            p_dest = self.np_random.integers(4)
            while p_loc == p_dest:
                p_dest = self.np_random.integers(4)
            state_tuple.extend([p_loc, p_dest])

        self.state = state_tuple
        return self.encode(self.state), {}

    def step(self, action):
        taxi_r, taxi_c = self.state[0], self.state[1]
        passengers = self.state[2:]
        reward = -1
        terminated = False

        if action == 0 and taxi_r < self.grid_size - 1: taxi_r += 1
        elif action == 1 and taxi_r > 0: taxi_r -= 1
        elif action == 2 and taxi_c < self.grid_size - 1: taxi_c += 1
        elif action == 3 and taxi_c > 0: taxi_c -= 1
        
        elif action == 4: # Pickup
            picked_up = False
            for i in range(self.num_passengers):
                p_loc = passengers[i*2]
                if p_loc < 4 and (taxi_r, taxi_c) == self.locs[p_loc]:
                    passengers[i*2] = 4
                    picked_up = True
                    break 
            if not picked_up: reward = -10
            
        elif action == 5: # Dropoff
            dropped_off = False
            for i in range(self.num_passengers):
                p_loc, p_dest = passengers[i*2], passengers[i*2+1]
                if p_loc == 4 and (taxi_r, taxi_c) == self.locs[p_dest]:
                    passengers[i*2] = p_dest 
                    dropped_off = True
                    reward = 20
                    break
            if not dropped_off: reward = -10

        self.state = [taxi_r, taxi_c] + passengers
        
        # Terminate if all passengers are at their destinations
        terminated = all(passengers[i*2] == passengers[i*2+1] for i in range(self.num_passengers))
        
        return self.encode(self.state), reward, terminated, False, {}

    def encode(self, state_tuple):
        return int(np.ravel_multi_index(state_tuple, self.state_bounds))

    def decode(self, state_int):
        return list(np.unravel_index(state_int, self.state_bounds))
    
    def render(self):
        taxi_r, taxi_c = self.state[0], self.state[1]
        passengers = self.state[2:]
        
        print("-" * (self.grid_size * 2 + 1))
        for r in range(self.grid_size):
            row_str = "|"
            for c in range(self.grid_size):
                cell = " "
                
                # 1. Mark Destinations (Magenta D) and Waiting Passengers (Blue P)
                for i in range(self.num_passengers):
                    p_loc, p_dest = passengers[i*2], passengers[i*2+1]
                    if (r, c) == self.locs[p_dest]:
                        cell = "\033[95mD\033[0m" 
                    if p_loc < 4 and (r, c) == self.locs[p_loc]:
                        cell = "\033[94mP\033[0m"
                
                # 2. Mark Taxi (Yellow T empty, Green X if carrying someone)
                if (r, c) == (taxi_r, taxi_c):
                    if any(passengers[i*2] == 4 for i in range(self.num_passengers)):
                        cell = "\033[92mX\033[0m" 
                    else:
                        cell = "\033[93mT\033[0m" 
                        
                row_str += cell + "|"
            print(row_str)
        print("-" * (self.grid_size * 2 + 1) + "\n")