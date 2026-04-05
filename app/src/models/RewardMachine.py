import os

from src.config import Configuration

class RewardMachine:
    def __init__(self, CONFIG: Configuration, file_name: str, default_reward=-1):
        self.file_path = os.path.join(CONFIG.MODELS_PATH, file_name)
        
        self.default_reward = default_reward
        self.initial_state = None
        self.final_state = None
        self._current_state = None
        self.states = dict() 

        lines = self._parse_lines()
        
        for line in lines:
            if line.startswith('i:'):
                self.initial_state = int(line.split(':')[1])
                continue
            if line.startswith('f:'):
                self.final_state = int(line.split(':')[1])
                continue

            u, u_next, condition, reward = line.split(';')
            u, u_next, condition, reward = (
                int(u), int(u_next), 
                tuple(condition.strip().split(',')), 
                float(reward)
            )

            if u not in self.states:
                self.states[u] = set()
            self.states[u].add((u_next, condition, reward))
        
        self._current_state = self.initial_state

    def _parse_lines(self):
        def _parse_line(line):
            clean = line.strip()
            pos = clean.find('#')
            if pos != -1:
                clean = clean[:pos].strip()
            return clean

        with open(self.file_path, 'r') as f:
            lines = [
                _parse_line(l)
                for l in f.readlines() 
                if l.strip() and not l.startswith('#')
            ]
        return lines

    def get_num_states(self):
        return len(self.states)
    
    def get_current_state(self):
        return self._current_state
    
    def reset(self):
        self._current_state = self.initial_state
    
    def simulate_step(self, start_u, events):
        """Simulates a step without changing the current state."""
        # Default to no transition and zero reward
        for next_u, condition, reward in self.states.get(start_u, []):
            holds = True
            for cond in condition:
                if cond.startswith("!"):
                    if cond[1:] in events: 
                        holds = False
                        break
                else:
                    if cond not in events: 
                        holds = False
                        break

            if holds:
                return next_u, reward, next_u == self.final_state

        return start_u, self.default_reward, start_u == self.final_state

    def step(self, events):
        """Advances the actual internal state of the RM."""
        res_u, res_reward, done = self.simulate_step(self._current_state, events)
        self._current_state = res_u
        return res_u, res_reward, done
