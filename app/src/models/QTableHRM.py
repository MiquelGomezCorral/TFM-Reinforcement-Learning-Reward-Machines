import numpy as np

from src.config import Configuration

from .QTable import QTable
from .HRM import HRM


class QTableHRM:
    """Two-level tabular HRM agent with a high-level policy and option actor."""

    def __init__(self, CONFIG: Configuration, env, rm_file) -> None:
        """
        Initialize high-level and actor Q-tables for a Reward Machine.

        The high-level table selects the next RM target. The actor table selects
        environment actions while pursuing the selected option.
        """
        self.CONFIG = CONFIG
        self.rm = RewardMachine(CONFIG, rm_file)
        # The final state has no outgoing transitions, so it is not an RM source state.
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
            initial_value=CONFIG.hrm_q_init,
        )
        self.actor = QTable(
            None,
            env.action_space.n,
            initial_value=CONFIG.hrm_q_init,
        )

    @staticmethod
    def _environment_state(state) -> object:
        """Convert vector observations into hashable table-key components."""
        if isinstance(state, np.ndarray):
            return tuple(state.reshape(-1).tolist())
        if isinstance(state, list):
            return tuple(state)
        return state

    def high_state(self, state, u) -> tuple:
        """Build the high-level Q-table key ``(environment_state, rm_state)``."""
        return self._environment_state(state), u

    def actor_state(self, state, u, target_u) -> tuple:
        """Build the actor Q-table key ``(environment_state, rm_state, target)``."""
        return self._environment_state(state), u, target_u

    def target_action(self, target_u) -> int:
        """Map an RM target state to its high-level Q-table action index."""
        return self._target_actions[target_u]

    def option_actions(self, u) -> tuple:
        """Return high-level action indices for options that start at RM state ``u``."""
        return tuple(self.target_action(target_u) for target_u in self.options[u])

    def max_high_value(self, state, u) -> float:
        """Return the largest high-level value among options valid at ``u``."""
        actions = self.option_actions(u)
        return float(np.max(self.high_level.values(self.high_state(state, u))[list(actions)]))

    def select_option(self, state, epsilon, u=None) -> int:
        """
        Select and activate an epsilon-greedy option from the valid targets at ``u``.

        Returns:
            The selected target RM state.
        """
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

    def print_size(self) -> None:
        """Print dynamic entry counts for both levels of the hierarchy."""
        self.high_level.print_size("High-level Q-table")
        self.actor.print_size("Actor Q-table")

    def get_rm_state(self) -> int:
        """Return the active Reward Machine state."""
        return self.rm.get_current_state()

    def reset_rm(self) -> None:
        """Reset the Reward Machine and clear any active option."""
        self.rm.reset()
        self.active_option = None

    def step_rm(self, events) -> tuple[int, float, bool]:
        """Advance the RM and end the active option after an RM transition."""
        current_u = self.get_rm_state()
        next_u, reward, done = self.rm.step(events)
        if next_u != current_u or done:
            self.active_option = None
        return next_u, reward, done

    def greedy_policy(self, state, u=None) -> int:
        """Select a greedy environment action under the active or newly selected option."""
        u = self.get_rm_state() if u is None else u
        if self.active_option not in self.options[u]:
            self.select_option(state, 0, u)

        return self.actor.greedy_policy(self.actor_state(state, u, self.active_option))

    def epsilon_greedy_policy(self, state, epsilon, env) -> int:
        """Select an epsilon-greedy environment action under the active option."""
        u = self.get_rm_state()

        if self.active_option not in self.options[u]:
            self.select_option(state, epsilon, u)

        return self.actor.epsilon_greedy_policy(
            self.actor_state(state, u, self.active_option),
            epsilon,
            env.action_space.sample,
        )

    def counterfactual_update(self, events: list, terminated: bool, state, action, new_state) -> None:
        """Update all actor tables for the same environment transition."""
        # Every option learns from the transition as a counterfactual experience.
        for counterfactual_u, targets in self.options.items():
            counterfactual_next_u, counterfactual_reward, _ = self.rm.simulate_step(
                counterfactual_u, events
            )
            option_done = terminated or counterfactual_next_u != counterfactual_u
            for target_u in targets:
                shaped_reward = self._option_reward(
                    counterfactual_reward,
                    counterfactual_u,
                    target_u,
                    counterfactual_next_u,
                    self.CONFIG.hrm_r_plus,
                    self.CONFIG.hrm_r_minus,
                )
                self.actor.update(
                    self.actor_state(state, counterfactual_u, target_u),
                    action,
                    shaped_reward,
                    self.actor_state(new_state, counterfactual_u, target_u),
                    option_done,
                    self.CONFIG.gamma,
                    self.CONFIG.learning_rate,
                )


    @staticmethod
    def _option_reward(base_reward, u, target_u, next_u, r_plus, r_minus):
        if target_u == u:
            return base_reward
        return base_reward + (r_plus if next_u == target_u else r_minus)
