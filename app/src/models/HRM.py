from .RewardMachine import RewardMachine


def option_reward(base_reward, target_u, next_u, option_done, r_plus, r_minus):
    if next_u == target_u:
        return base_reward + r_plus
    if option_done:
        return base_reward + r_minus
    return base_reward


class HRM:
    def __init__(self, config, rm_file):
        self.rm = RewardMachine(config, rm_file)
        self.rm_states = tuple(sorted(self.rm.states))
        self.target_states = tuple(sorted({*self.rm_states, self.rm.final_state}))
        self._target_actions = {
            target: action for action, target in enumerate(self.target_states)
        }
        self.options = {
            u: tuple(sorted({target for target, _, _ in transitions if target != u}))
            for u, transitions in self.rm.states.items()
        }
        empty_options = [u for u, targets in self.options.items() if not targets]
        if empty_options:
            raise ValueError(f"HRM states without non-self options: {empty_options}")
        self.active_option = None

    def target_action(self, target_u):
        return self._target_actions[target_u]

    def option_actions(self, u):
        return tuple(self.target_action(target_u) for target_u in self.options[u])

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
