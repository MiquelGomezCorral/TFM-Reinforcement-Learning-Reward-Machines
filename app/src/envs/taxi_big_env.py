import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame # New import


class MultiTaxiEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size=5,
        num_passengers=2,
        observation_mode="discrete",
        render_mode=None,
    ):
        self.grid_size = grid_size
        self.num_passengers = num_passengers
        self.observation_mode = observation_mode
        self.render_mode = render_mode

        if grid_size == 5:
            self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        elif grid_size == 10:
            self.locs = [(0, 0), (0, 9), (9, 0), (9, 8)]
        else:
            raise ValueError("grid_size must be 5 or 10")

        self.state_bounds = [grid_size, grid_size] + [5, 4] * num_passengers
        if observation_mode == "discrete":
            self.observation_space = spaces.Discrete(int(np.prod(self.state_bounds)))
        elif observation_mode == "relative":
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7 * num_passengers,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported observation_mode: {observation_mode}")
        self.action_space = spaces.Discrete(6)
        self.state = None
        
        # Pygame variables (lazy init in render)
        self.window_size = 512
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None
        self.p_colors = [(255, 0, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128), (0, 255, 255)]

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
        raw_state = self.encode(self.state)
        return self._observation(), {"raw_state": raw_state}

    def step(self, action):
        taxi_r, taxi_c = self.state[0], self.state[1]
        passengers = self.state[2:]
        reward = -1
        terminated = False

        if action == 0 and taxi_r < self.grid_size - 1: taxi_r += 1
        elif action == 1 and taxi_r > 0: taxi_r -= 1
        elif action == 2 and taxi_c < self.grid_size - 1: taxi_c += 1
        elif action == 3 and taxi_c > 0: taxi_c -= 1
        
        # Inside the step(self, action) method
        elif action == 4:
            picked_up = False
            for i in range(self.num_passengers):
                p_loc = passengers[i*2]
                p_dest = passengers[i*2+1] # Added destination check
                
                # Check that they are not already at their destination!
                if p_loc < 4 and p_loc != p_dest and (taxi_r, taxi_c) == self.locs[p_loc]:
                    passengers[i*2] = 4
                    picked_up = True
            if not picked_up: reward = -10

        elif action == 5:
            dropped_off = False
            for i in range(self.num_passengers):
                p_loc, p_dest = passengers[i*2], passengers[i*2+1]
                if p_loc == 4 and (taxi_r, taxi_c) == self.locs[p_dest]:
                    passengers[i*2] = p_dest 
                    dropped_off = True
            if dropped_off:
                reward = 20
            else:
                reward = -10

        self.state = [taxi_r, taxi_c] + passengers
        terminated = all(passengers[i*2] == passengers[i*2+1] for i in range(self.num_passengers))
        
        raw_state = self.encode(self.state)
        return self._observation(), reward, terminated, False, {"raw_state": raw_state}

    def _observation(self):
        if self.observation_mode == "discrete":
            return self.encode(self.state)

        taxi_r, taxi_c = self.state[:2]
        scale = self.grid_size - 1
        observation = []
        for i in range(self.num_passengers):
            passenger_location, destination = self.state[2 + i * 2:4 + i * 2]
            destination_r, destination_c = self.locs[destination]

            if passenger_location == 4:
                passenger_r, passenger_c = taxi_r, taxi_c
                status = (0.0, 1.0, 0.0)
            elif passenger_location == destination:
                passenger_r, passenger_c = destination_r, destination_c
                status = (0.0, 0.0, 1.0)
            else:
                passenger_r, passenger_c = self.locs[passenger_location]
                status = (1.0, 0.0, 0.0)

            observation.extend([
                (passenger_r - taxi_r) / scale,
                (passenger_c - taxi_c) / scale,
                (destination_r - taxi_r) / scale,
                (destination_c - taxi_c) / scale,
                *status,
            ])

        return np.asarray(observation, dtype=np.float32)

    def encode(self, state_tuple):
        return int(np.ravel_multi_index(state_tuple, self.state_bounds))

    def decode(self, state_int):
        return list(np.unravel_index(state_int, self.state_bounds))
    
    def render(self):
        if self.render_mode is None:
            return

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # White background

        taxi_r, taxi_c = self.state[0], self.state[1]
        passengers = self.state[2:]
        cs = self.cell_size

        # Draw Destinations (Unique color, smaller nested squares if overlapping)
        for i in range(self.num_passengers):
            dr, dc = self.locs[passengers[i*2+1]]
            color = self.p_colors[i % len(self.p_colors)]
            inset = 10 + (i * 4) 
            pygame.draw.rect(canvas, color, pygame.Rect(dc*cs + inset, dr*cs + inset, cs - 2*inset, cs - 2*inset), 3)

        # Draw Taxi (Yellow if empty, Green if full)
        taxi_full = any(passengers[i*2] == 4 for i in range(self.num_passengers))
        taxi_color = (0, 255, 0) if taxi_full else (255, 255, 0)
        pygame.draw.rect(canvas, taxi_color, pygame.Rect(taxi_c*cs + 15, taxi_r*cs + 15, cs - 30, cs - 30))

        # Draw Passengers (Unique color, spaced side-by-side)
        for i in range(self.num_passengers):
            p_loc = passengers[i*2]
            color = self.p_colors[i % len(self.p_colors)]
            if p_loc < 4:
                pr, pc = self.locs[p_loc]
                spacing = cs // (self.num_passengers + 1)
                px = pc * cs + spacing * (i + 1)
                py = pr * cs + cs // 2
                pygame.draw.circle(canvas, color, (px, py), cs // 6)

        # Draw Grid Lines
        for x in range(0, self.window_size + 1, cs):
            pygame.draw.line(canvas, (200, 200, 200), (x, 0), (x, self.window_size))
            pygame.draw.line(canvas, (200, 200, 200), (0, x), (self.window_size, x))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
