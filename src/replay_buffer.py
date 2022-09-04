import numpy as np
from typing import List, Tuple
from collections import deque
from dataclasses import dataclass

from src.envs.gec_env import TOKENS, ACTIONS


@dataclass(frozen=True)
class BatchSample:
    __slots__ = ["states", "actions", "rewards", "next_states", "is_terminals"]
    states: Tuple[TOKENS]
    actions: Tuple[ACTIONS]
    rewards: Tuple[float]
    next_states: Tuple[TOKENS]
    is_terminals: Tuple[bool]

    def __len__(self) -> int:
        return len(self.is_terminals)


class ReplayBuffer:
    def __init__(self, max_len: int = 100_000, pop_percent: float = 0.10):
        self.max_len = max_len
        self.buffer = deque(maxlen=max_len)
        self.pop_size = round(pop_percent * max_len)

    def __len__(self):
        return len(self.buffer)

    def add(self, state: TOKENS, action: ACTIONS, reward: float, next_state: TOKENS, is_terminal: bool):
        self.buffer.append((state, action, reward, next_state, is_terminal))
        if len(self) >= self.max_len:
            self.pop_oldest()

    def pop_oldest(self):
        for _ in range(self.pop_size):
            self.buffer.pop()

    def sample(self, batch_size: int = 1) -> BatchSample:
        buffer_size = len(self)
        if batch_size > buffer_size:
            batch_size = buffer_size
        indexes = np.random.choice(buffer_size, size=batch_size, replace=False)
        item_generator = (self.buffer[i] for i in indexes)
        states, actions, rewards, next_states, is_terminals = zip(*item_generator)
        batch = BatchSample(states, actions, rewards, next_states, is_terminals)
        return batch

    def clear(self):
        self.buffer.clear()
