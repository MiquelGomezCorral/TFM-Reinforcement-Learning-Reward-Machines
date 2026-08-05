import random

import numpy as np

from src.utils import compute_epsilon

from .DQN import DQN
from .HRM import HRM, option_reward


class DQNHRM(HRM):
    def __init__(self, CONFIG, env, rm_file):
        if len(env.observation_space.shape) != 1:
            raise ValueError("DQNHRM requires one-dimensional vector observations")

        super().__init__(CONFIG, rm_file)
        self._rm_indices = {u: index for index, u in enumerate(self.rm_states)}
        self._valid_option_states = None

        observation_size = env.observation_space.shape[0]
        self.high_level = self._new_dqn(
            CONFIG,
            observation_size + len(self.rm_states),
            len(self.target_states),
        )
        self.actor = self._new_dqn(
            CONFIG,
            observation_size + len(self.rm_states) + len(self.target_states),
            env.action_space.n,
            sum(map(len, self.options.values())),
        )

    @staticmethod
    def _new_dqn(CONFIG, input_size, action_size, batch_multiplier=1):
        return DQN(
            input_size=input_size,
            action_size=action_size,
            batch_size=CONFIG.dqn_batch_size * batch_multiplier,
            replay_capacity=CONFIG.dqn_replay_capacity * batch_multiplier,
            learning_rate=CONFIG.dqn_learning_rate,
            gamma=CONFIG.gamma,
            hidden_size=CONFIG.dqn_hidden_size,
            tau=CONFIG.dqn_tau,
            gradient_clip=CONFIG.dqn_gradient_clip,
        )

    def reset_rm(self):
        super().reset_rm()
        self._valid_option_states = None

    def _rm_state(self, u):
        state = np.zeros(len(self.rm_states), dtype=np.float32)
        state[self._rm_indices[u]] = 1.0
        return state

    def _target_state(self, target_u):
        state = np.zeros(len(self.target_states), dtype=np.float32)
        state[self.target_action(target_u)] = 1.0
        return state

    @staticmethod
    def _environment_state(state):
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def high_state(self, state, u):
        return np.concatenate((self._environment_state(state), self._rm_state(u)))

    def actor_state(self, state, u, target_u):
        return np.concatenate((
            self._environment_state(state),
            self._rm_state(u),
            self._target_state(target_u),
        ))

    def max_high_value(self, state, u):
        actions = self.option_actions(u)
        values = self.high_level.q_values(self.high_state(state, u), target=True)
        return float(np.max(values[list(actions)]))

    def select_option(self, state, epsilon, u=None):
        u = self.get_rm_state() if u is None else u
        targets = self.options[u]
        if random.random() < epsilon:
            target_u = random.choice(targets)
        else:
            actions = self.option_actions(u)
            values = self.high_level.q_values(self.high_state(state, u))[list(actions)]
            target_u = targets[int(np.argmax(values))]
        self.active_option = target_u
        return target_u

    def print_size(self):
        print(" - High-level", end="")
        self.high_level.print_size()
        print(" - Actor", end="")
        self.actor.print_size()

    def greedy_policy(self, state, u=None):
        u = self.get_rm_state() if u is None else u
        if self.active_option not in self.options[u]:
            self.select_option(state, 0, u)
        return self.actor.greedy_policy(self.actor_state(state, u, self.active_option))

    def epsilon_greedy_policy(self, state, epsilon, sample_action=None):
        u = self.get_rm_state()
        if self.active_option not in self.options[u]:
            self.select_option(state, epsilon, u)
        return self.actor.epsilon_greedy_policy(
            self.actor_state(state, u, self.active_option),
            epsilon,
            sample_action,
        )

    def training_epsilon(self, _episode, total_steps):
        return compute_epsilon(
            self.config.min_epsilon,
            self.config.max_epsilon,
            total_steps,
            1 / self.config.dqn_epsilon_decay_steps,
        )

    def counterfactual_update(
        self,
        events,
        terminated,
        state,
        action,
        new_state,
        env_reward,
        invalid_action,
    ):
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
                    env_reward if invalid_action else counterfactual_reward,
                    target_u,
                    counterfactual_next_u,
                    option_done,
                    self.config.hrm_r_plus,
                    self.config.hrm_r_minus,
                )
                self.actor.remember(
                    self.actor_state(state, counterfactual_u, target_u),
                    action,
                    shaped_reward,
                    None if option_done else self.actor_state(
                        new_state, counterfactual_u, target_u
                    ),
                    option_done,
                )
        self._valid_option_states = reachable_states

    def update_high_level(self, state, action, target, _new_state, _next_u):
        self.high_level.update(state, action, target, None, True, optimize=False)

    def optimize_training_step(self, total_steps):
        if (
            total_steps >= self.config.dqn_learning_starts
            and total_steps % self.config.dqn_optimize_interval == 0
        ):
            self.actor.optimize()
            self.high_level.optimize()
