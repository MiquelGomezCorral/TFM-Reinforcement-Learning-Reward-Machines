import numpy as np

from src.config import Configuration

from .RewardMachine import RewardMachine


class QTable:
    """Tabular action-value function with dynamic or preallocated storage."""

    def __init__(self, state_space, action_space, dynamic: bool = True, initial_value: float = 0) -> None:
        """
        Initialize a Q-table.

        Args:
            state_space: Number of states for a static table; ignored for dynamic tables.
            action_space: Number of actions represented by each table entry.
            dynamic: Store only visited states when True.
            initial_value: Q-value assigned to new entries.
        """
        self.action_space = action_space
        self.dynamic = dynamic
        self.initial_value = initial_value
        self._table = (
            {} if dynamic
            else np.full((state_space, action_space), initial_value, dtype=float)
        )

    def _state_key(self, state) -> object:
        """Convert array observations into dictionary keys."""
        return tuple(state) if isinstance(state, np.ndarray) else state

    def values(self, state) -> np.ndarray:
        """
        Return the action values for a state, allocating dynamic entries on demand.

        Args:
            state: Environment or composite state used as the table key.

        Returns:
            NumPy array containing one Q-value per action.
        """
        if not self.dynamic:
            return self._table[state]

        key = self._state_key(state)
        if key not in self._table:
            self._table[key] = np.full(self.action_space, self.initial_value, dtype=float)
        return self._table[key]

    def print_size(self, label: str = "Q-Table") -> None:
        """Print the number of dynamic entries or the static array shape."""
        if self.dynamic:
            size = f"{len(self._table)} states"
        else:
            size = self._table.shape
        print(f" - {label}: {size}")

    def greedy_policy(self, state) -> int:
        """Return the action with the greatest Q-value for a state."""
        return int(np.argmax(self.values(state)))

    def epsilon_greedy_policy(self, state, epsilon, sample_action=None) -> int:
        """
        Choose a greedy action or explore with probability ``epsilon``.

        Args:
            state: State used to select an action.
            epsilon: Exploration probability.
            sample_action: Optional environment action sampler.
        """
        if np.random.random() > epsilon:
            return self.greedy_policy(state)
        return sample_action() if sample_action else int(np.random.randint(self.action_space))

    def update(self, state, action, reward, new_state, done, gamma, learning_rate, next_q_table=None) -> None:
        """
        Apply a one-step Q-learning update.

        ``next_q_table`` supports bootstrapping from a different table, as used by
        Reward Machine transitions.
        """
        q_values = self.values(state)
        next_q_table = self if next_q_table is None else next_q_table
        future = 0 if done else gamma * np.max(next_q_table.values(new_state))
        q_values[action] += learning_rate * (reward + future - q_values[action])


class QTableRM:
    """Q-learning agent indexed by Reward Machine state."""

    def __init__(self, CONFIG: Configuration, env, rm_file: str = None, dynamic: bool = True) -> None:
        """
        Initialize one Q-table per Reward Machine source state.

        Args:
            CONFIG: Training configuration and RM model path.
            env: Gymnasium environment supplying state and action spaces.
            rm_file: Reward Machine definition; omit it for ordinary Q-learning.
            dynamic: Store only visited environment states when True.
        """
        print(f"The QTable is {'dynamic' if dynamic else 'static'}")
        self.rm = RewardMachine(CONFIG, rm_file) if rm_file else None
        self.state_space = None if dynamic else env.observation_space.n
        self.action_space = env.action_space.n
        rm_states = self.rm.states if self.rm else [0]
        self._q_tables = {u: self._new_q_table() for u in rm_states}

    def _new_q_table(self) -> QTable:
        """Create a Q-table matching this agent's environment action space."""
        return QTable(self.state_space, self.action_space, dynamic=self.state_space is None)

    def _q_table(self, u) -> QTable:
        """Return the Q-table for RM state ``u``, creating it if necessary."""
        if u not in self._q_tables:
            self._q_tables[u] = self._new_q_table()
        return self._q_tables[u]

    def print_size(self) -> None:
        """Print the stored state count for every RM-specific Q-table."""
        for u, q_table in self._q_tables.items():
            q_table.print_size(f"RM state {u} Q-table")

    def get_rm_state(self) -> int:
        """Return the active RM state, or zero when no RM is configured."""
        return self.rm.get_current_state() if self.rm else 0

    def step_rm(self, events) -> tuple[int, float, bool]:
        """Advance the Reward Machine from observed propositions."""
        if self.rm:
            return self.rm.step(events)
        return 0, 0, False

    def reset_rm(self) -> None:
        """Reset the configured Reward Machine at the start of an episode."""
        if self.rm:
            self.rm.reset()

    def greedy_policy(self, state, u=None) -> int:
        """Select the greedy environment action for an RM state."""
        if u is None:
            u = self.get_rm_state()
        return self._q_table(u).greedy_policy(state)

    def epsilon_greedy_policy(self, state, epsilon, env) -> int:
        """Select an epsilon-greedy environment action in the active RM state."""
        return self._q_table(self.get_rm_state()).epsilon_greedy_policy(
            state, epsilon, env.action_space.sample
        )

    def update(
        self, state, action, env_reward, raw_state, new_state, new_state_parse,
        gamma, learning_rate, env, get_propositions, terminated=False, use_crm=False,
    ) -> bool:
        """
        Update Q-values from one environment transition.

        When CRM is enabled, the same transition updates every RM-state table
        counterfactually; otherwise only the active RM-state table is updated.
        """
        current_u = self.get_rm_state()
        rm_done = False

        if self.rm:
            events = get_propositions(env, raw_state, action, new_state)

            target_u, reward, rm_done = self.step_rm(events)
            if use_crm:
                for u in self.rm.states:
                    t_u, r_u, d_u = self.rm.simulate_step(u, events)
                    done = terminated or d_u
                    self._q_table(u).update(
                        state, action, r_u, new_state_parse, done, gamma, learning_rate,
                        None if done else self._q_table(t_u),
                    )
            else:
                done = terminated or rm_done
                self._q_table(current_u).update(
                    state, action, reward, new_state_parse, done, gamma, learning_rate,
                    None if done else self._q_table(target_u),
                )
        else:
            self._q_table(current_u).update(
                state, action, env_reward, new_state_parse, terminated, gamma, learning_rate
            )

        return rm_done
