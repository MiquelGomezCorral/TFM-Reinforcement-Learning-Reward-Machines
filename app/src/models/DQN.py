import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim

from .DQNNetwork import DQNNetwork


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray | None


class ReplayMemory:
    def __init__(self, capacity):
        self._transitions = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self._transitions.append(Transition(state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self._transitions, batch_size)

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
        self.memory = ReplayMemory(replay_capacity)

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

    def epsilon_greedy_policy(self, state, epsilon, sample_action=None):
        if random.random() > epsilon:
            return self.greedy_policy(state)
        return sample_action() if sample_action else random.randrange(self.action_size)

    def update(self, state, action, reward, next_state, terminal):
        self.remember(state, action, reward, next_state, terminal)
        return self.optimize()

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
