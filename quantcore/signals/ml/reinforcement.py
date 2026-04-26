"""Reinforcement learning for trading: DQN, PPO, SAC + custom TradingEnv (Gym-like).

Note: Full implementations require gym/torch. This module provides
simplified numpy-based environments and Q-learning as a foundation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Trading Environment
# ---------------------------------------------------------------------------


@dataclass
class TradingEnvState:
    """State of the trading environment at a given step."""

    position: int  # -1, 0, +1 (short, flat, long)
    entry_price: float
    unrealized_pnl: float
    step: int
    portfolio_value: float


class TradingEnv:
    """Simple gym-like trading environment for RL.

    Observation: [returns_1..n, position, unrealized_pnl_ratio]
    Actions: 0=short, 1=flat, 2=long
    Reward: realized P&L after transaction costs.
    """

    def __init__(
        self,
        prices: NDArray,
        transaction_cost_bps: float = 5.0,
        lookback: int = 10,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.prices = prices
        self.returns = np.zeros(len(prices), dtype=float)
        self.returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        self.tc = transaction_cost_bps / 10000.0
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.reset()

    def reset(self) -> NDArray:
        self.position = 0
        self.entry_price = 0.0
        self._step_idx = self.lookback
        self.portfolio_value = self.initial_capital
        self.total_pnl = 0.0
        return self._get_obs()

    def _get_obs(self) -> NDArray:
        """Get observation vector: recent returns + position + unrealized P&L."""
        ret_window = self.returns[self._step_idx - self.lookback:self._step_idx]
        ul_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            ul_pnl = self.position * (self.prices[self._step_idx] - self.entry_price) / self.entry_price
        return np.concatenate([ret_window, [float(self.position), ul_pnl]])

    @property
    def obs_dim(self) -> int:
        return self.lookback + 2

    @property
    def n_actions(self) -> int:
        return 3  # Short, Flat, Long

    def step(self, action: int) -> tuple[NDArray, float, bool]:
        """Execute action, return (obs, reward, done)."""
        target_position = action - 1  # 0→-1, 1→0, 2→+1

        reward = 0.0
        price = self.prices[self._step_idx]

        # Realize P&L on position change
        if target_position != self.position:
            if self.position != 0 and self.entry_price > 0:
                realized = self.position * (price - self.entry_price) / self.entry_price
                reward += realized
                self.total_pnl += realized

            # Transaction cost
            reward -= self.tc
            self.total_pnl -= self.tc

            self.position = target_position
            self.entry_price = price if target_position != 0 else 0.0

        # Mark-to-market on unchanged position
        elif self.position != 0 and self.entry_price > 0:
            mtm = self.position * (price - self.entry_price) / self.entry_price
            reward += mtm * 0.01  # Small incremental reward for holding

        self._step_idx += 1
        done = self._step_idx >= len(self.prices) - 1

        obs = self._get_obs() if not done else np.zeros(self.obs_dim)
        return obs, reward, done


# ---------------------------------------------------------------------------
# Q-Learning Agent (Tabular / Simple)
# ---------------------------------------------------------------------------


class QLearningAgent:
    """Simple Q-learning agent for the trading environment."""

    def __init__(
        self,
        obs_bins: int = 10,
        n_actions: int = 3,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
    ) -> None:
        self.obs_bins = obs_bins
        self.n_actions = n_actions
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table: dict[tuple, NDArray] = {}

    def _discretize(self, obs: NDArray) -> tuple:
        """Discretize continuous observation into bins."""
        binned = []
        for val in obs:
            if np.isnan(val):
                binned.append(0)
            else:
                idx = int(np.clip(val * self.obs_bins, -self.obs_bins, self.obs_bins))
                binned.append(idx)
        return tuple(binned)

    def act(self, obs: NDArray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        state = self._discretize(obs)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def learn(self, obs: NDArray, action: int, reward: float, next_obs: NDArray, done: bool) -> None:
        """Update Q-value."""
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)

        if done:
            target = reward
        else:
            target = reward + self.discount * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_qlearning(
    env: TradingEnv,
    n_episodes: int = 100,
    learning_rate: float = 0.1,
    epsilon_start: float = 0.5,
    epsilon_end: float = 0.01,
) -> QLearningAgent:
    """Train Q-learning agent on trading environment."""
    agent = QLearningAgent(
        obs_bins=5,
        n_actions=env.n_actions,
        learning_rate=learning_rate,
        epsilon=epsilon_start,
    )

    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0

        # Linear epsilon decay
        agent.epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / n_episodes)

        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

    return agent
