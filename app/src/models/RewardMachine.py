import os

from src.config import Configuration


class RewardMachine:
    """Deterministic Reward Machine loaded from a text transition definition."""

    def __init__(self, CONFIG: Configuration, file_name: str) -> None:
        """
        Load and validate a Reward Machine.

        Args:
            CONFIG: Configuration providing the models directory.
            file_name: RM definition file relative to ``CONFIG.MODELS_PATH``.
        """
        self.file_path = os.path.join(CONFIG.MODELS_PATH, file_name)
        self.default_reward = 0
        self.initial_state = None
        self.final_state = None
        self._current_state = None
        self.states = {}

        lines = self._parse_lines()
        self._process_lines(lines)
        self._validate_targets()

    def _parse_lines(self) -> list[str]:
        """Read nonempty, uncommented lines from the RM definition file."""
        lines = []
        with open(self.file_path, "r", encoding="utf-8") as file:
            for line in file:
                clean = line.split("#", 1)[0].strip()
                if clean:
                    lines.append(clean)
        return lines

    def _process_lines(self, lines) -> None:
        """Parse RM headers and transition rows into internal state structures."""
        has_initial = False
        has_final = False

        for line in lines:
            if line.startswith('i:'):
                if has_initial:
                    continue
                self.initial_state = int(line.split(':')[1])
                self._current_state = self.initial_state
                has_initial = True
                continue

            if line.startswith('f:'):
                if has_final:
                    continue
                self.final_state = int(line.split(':')[1])
                has_final = True
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

    def _validate_targets(self) -> None:
        """Reject referenced non-final states that have no outgoing transitions."""
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

    def get_num_states(self) -> int:
        """Return the number of RM states with outgoing transitions."""
        return len(self.states)

    def get_current_state(self) -> int:
        """Return the currently active RM state."""
        return self._current_state

    def reset(self) -> None:
        """Reset the active RM state to the configured initial state."""
        self._current_state = self.initial_state

    def simulate_step(self, start_u, events) -> tuple[int, float, bool]:
        """
        Simulate an RM transition without changing the active state.

        Returns:
            Tuple containing next RM state, RM reward, and terminal flag.
        """
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

    def step(self, events) -> tuple[int, float, bool]:
        """Advance the active RM state from observed propositions."""
        res_u, res_reward, done = self.simulate_step(self._current_state, events)
        self._current_state = res_u
        return res_u, res_reward, done
