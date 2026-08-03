import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray | None


class DQNNetwork(nn.Module):
    def __init__(self, input_size, action_size, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        return self.layers(state)


class ReplayMemory:
    def __init__(self, capacity, rewarding_fraction=0.25):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.rewarding_fraction = rewarding_fraction
        self._transitions = []
        self._position = 0
        self._rewarding_slots = []
        self._rewarding_indices = {}

    def push(self, state, action, reward, next_state):
        transition = Transition(state, action, reward, next_state)
        if len(self._transitions) == self.capacity:
            self._remove_rewarding_slot(self._position)
            self._transitions[self._position] = transition
        else:
            self._transitions.append(transition)
        if reward > 0:
            self._rewarding_indices[self._position] = len(self._rewarding_slots)
            self._rewarding_slots.append(self._position)
        self._position = (self._position + 1) % self.capacity

    def _remove_rewarding_slot(self, slot):
        index = self._rewarding_indices.pop(slot, None)
        if index is None:
            return
        last_slot = self._rewarding_slots.pop()
        if index < len(self._rewarding_slots):
            self._rewarding_slots[index] = last_slot
            self._rewarding_indices[last_slot] = index

    def sample(self, batch_size):
        slots = random.sample(range(len(self._transitions)), batch_size)
        rewarding_target = (
            max(1, int(batch_size * self.rewarding_fraction))
            if self.rewarding_fraction > 0 else 0
        )
        rewarding_count = min(len(self._rewarding_slots), batch_size, rewarding_target)
        selected = set(slots)
        missing = rewarding_count - sum(slot in self._rewarding_indices for slot in slots)
        replace_index = 0
        while missing > 0:
            rewarding_slot = random.choice(self._rewarding_slots)
            if rewarding_slot in selected:
                continue
            while slots[replace_index] in self._rewarding_indices:
                replace_index += 1
            selected.remove(slots[replace_index])
            selected.add(rewarding_slot)
            slots[replace_index] = rewarding_slot
            missing -= 1
        return [self._transitions[slot] for slot in slots]

    def __len__(self):
        return len(self._transitions)


class DQN:
    def __init__(
        self,
        input_size,
        action_size,
        batch_size,
        replay_capacity,
        learning_rate,
        gamma,
        hidden_size,
        tau,
        gradient_clip,
        rewarding_fraction=0.25,
    ):
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.gradient_clip = gradient_clip
        self.device = self._device()
        self.policy_net = DQNNetwork(input_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNNetwork(input_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(replay_capacity, rewarding_fraction)

    @staticmethod
    def _device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _state_tensor(self, state):
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)

    def print_size(self):
        parameters = sum(parameter.numel() for parameter in self.policy_net.parameters())
        print(f" - DQN parameters: {parameters}, replay entries: {len(self.memory)}")

    def greedy_policy(self, state):
        with torch.no_grad():
            return int(self.policy_net(self._state_tensor(state)).argmax(dim=1).item())

    def q_values(self, state, target=False):
        network = self.target_net if target else self.policy_net
        with torch.no_grad():
            return network(self._state_tensor(state)).squeeze(0).cpu().numpy()

    def epsilon_greedy_policy(self, state, epsilon, sample_action=None):
        if random.random() > epsilon:
            return self.greedy_policy(state)
        return sample_action() if sample_action else random.randrange(self.action_size)

    def update(self, state, action, reward, next_state, terminal, optimize=True):
        self.remember(state, action, reward, next_state, terminal)
        return self.optimize() if optimize else None

    def remember(self, state, action, reward, next_state, terminal):
        state = np.array(state, dtype=np.float32, copy=True)
        next_state = None if terminal else np.array(next_state, dtype=np.float32, copy=True)
        self.memory.push(state, action, reward, next_state)

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        states = torch.as_tensor(
            np.stack([transition.state for transition in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [transition.action for transition in transitions],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        state_action_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        non_terminal_indices = [
            index for index, transition in enumerate(transitions)
            if transition.next_state is not None
        ]
        if non_terminal_indices:
            next_states = torch.as_tensor(
                np.stack([transitions[index].next_state for index in non_terminal_indices]),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                next_state_values[non_terminal_indices] = self.target_net(next_states).max(dim=1).values

        expected_values = rewards + self.gamma * next_state_values
        loss = nn.SmoothL1Loss()(state_action_values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        self._soft_update_target()
        return float(loss.item())

    def _soft_update_target(self):
        for target_parameter, policy_parameter in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_parameter.data.mul_(1 - self.tau).add_(policy_parameter.data, alpha=self.tau)
