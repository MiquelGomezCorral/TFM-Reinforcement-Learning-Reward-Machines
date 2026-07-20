from .RewardMachine import RewardMachine


def option_reward(base_reward, u, target_u, next_u, r_plus, r_minus):
    if target_u == u:
        return base_reward
    return base_reward + (r_plus if next_u == target_u else r_minus)


class HRM:
    def __init__(self, config, rm_file):
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
