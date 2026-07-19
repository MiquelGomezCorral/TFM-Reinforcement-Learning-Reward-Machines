import os

from src.config import Configuration

class RewardMachine:
    def __init__(self, CONFIG: Configuration, file_name: str):
        self.file_path = os.path.join(CONFIG.MODELS_PATH, file_name)
        self.default_reward = 0
        self.initial_state = None
        self.final_state = None
        self._current_state = None
        self.states = {}

        lines = self._parse_lines()
        
        for line in lines:
            if line.startswith('i:'):
                self.initial_state = int(line.split(':')[1])
                continue
            if line.startswith('f:'):
                self.final_state = int(line.split(':')[1])
                continue
            if line.startswith('r:'):
                self.default_reward = float(line.split(':')[1])
                continue

            u, u_next, condition, reward = line.split(';')
            u, u_next, condition, reward = (
                int(u), int(u_next), 
                tuple(c.strip() for c in condition.strip().split(',')), 
                float(reward)
            )

            self.states.setdefault(u, []).append((u_next, condition, reward))

        if self.initial_state is None or self.final_state is None:
            raise ValueError("Reward Machine requires initial and final states")

        targets = {
            target
            for transitions in self.states.values()
            for target, _, _ in transitions
        }
        invalid_states = (targets | {self.initial_state}) - self.states.keys() - {self.final_state}
        if invalid_states:
            raise ValueError(
                f"Non-final RM states without transitions: {sorted(invalid_states)}"
            )

        self._current_state = self.initial_state

    def _parse_lines(self):
        lines = []
        with open(self.file_path, "r", encoding="utf-8") as file:
            for line in file:
                clean = line.split("#", 1)[0].strip()
                if clean:
                    lines.append(clean)
        return lines

    def get_num_states(self):
        return len(self.states)
    
    def get_current_state(self):
        return self._current_state
    
    def reset(self):
        self._current_state = self.initial_state
    
    def simulate_step(self, start_u, events):
        """Simulates a step without changing the current state."""
        matches = []
        for next_u, condition, reward in self.states.get(start_u, []):
            if all(
                cond[1:] not in events if cond.startswith("!") else cond in events
                for cond in condition
            ):
                matches.append((next_u, reward))

        if len(matches) > 1:
            raise ValueError(f"Ambiguous transitions from RM state {start_u}: {matches}")
        if matches:
            next_u, reward = matches[0]
            return next_u, reward, next_u == self.final_state

        return start_u, self.default_reward, start_u == self.final_state

    def step(self, events):
        """Advances the actual internal state of the RM."""
        res_u, res_reward, done = self.simulate_step(self._current_state, events)
        self._current_state = res_u
        return res_u, res_reward, done
