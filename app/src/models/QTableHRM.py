import numpy as np

from src.config import Configuration

from .QTable import QTable
from .HRM import HRM, option_reward


class QTableHRM(HRM):
    """Two-level tabular HRM agent with a high-level policy and option actor."""

    def __init__(self, config: Configuration, env, rm_file) -> None:
        """
        Initialize high-level and actor Q-tables for a Reward Machine.

        The high-level table selects the next RM target. The actor table selects
        environment actions while pursuing the selected option.
        """
        super().__init__(config, rm_file)
        self.high_level = QTable(
            None,
            len(self.target_states),
            initial_value=config.hrm_q_init,
        )
        self.actor = QTable(
            None,
            env.action_space.n,
            initial_value=config.hrm_q_init,
        )
        self._valid_option_states = None

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

    def reset_rm(self) -> None:
        super().reset_rm()
        self._valid_option_states = None

    def greedy_policy(self, state, u=None) -> int:
        """Select a greedy environment action under the active or newly selected option."""
        u = self.get_rm_state() if u is None else u
        if self.active_option not in self.options[u]:
            self.select_option(state, 0, u)

        return self.actor.greedy_policy(self.actor_state(state, u, self.active_option))

    def epsilon_greedy_policy(self, state, epsilon, sample_action=None) -> int:
        """Select an epsilon-greedy environment action under the active option."""
        u = self.get_rm_state()

        if self.active_option not in self.options[u]:
            self.select_option(state, epsilon, u)

        return self.actor.epsilon_greedy_policy(
            self.actor_state(state, u, self.active_option),
            epsilon,
            sample_action,
        )

    def counterfactual_update(
        self,
        events: list,
        terminated: bool,
        state,
        action,
        new_state,
        _env_reward,
        _invalid_action,
    ) -> None:
        """Update all actor tables for the same environment transition."""
        reachable_states = set()
        for counterfactual_u, targets in self.options.items():
            counterfactual_next_u, counterfactual_reward, _ = self.rm.simulate_step(
                counterfactual_u, events
            )
            reachable_states.add(counterfactual_next_u)
            if (
                self._valid_option_states is not None
                and counterfactual_u not in self._valid_option_states
            ):
                continue
            option_done = terminated or counterfactual_next_u != counterfactual_u
            for target_u in targets:
                shaped_reward = option_reward(
                    counterfactual_reward,
                    target_u,
                    counterfactual_next_u,
                    option_done,
                    self.config.hrm_r_plus,
                    self.config.hrm_r_minus,
                )
                self.actor.update(
                    self.actor_state(state, counterfactual_u, target_u),
                    action,
                    shaped_reward,
                    self.actor_state(new_state, counterfactual_u, target_u),
                    option_done,
                    self.config.gamma,
                    self.config.learning_rate,
                )
        self._valid_option_states = reachable_states

    def update_high_level(self, state, action, target, new_state, next_u) -> None:
        self.high_level.update(
            state,
            action,
            target,
            self.high_state(new_state, next_u),
            True,
            self.config.gamma,
            self.config.learning_rate,
        )
