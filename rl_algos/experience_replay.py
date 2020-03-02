import torch
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
device = 'cpu'


class ExperienceReplayBuffer():
    def __init__(self, buffer_size: int, batch_size: int):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add_experience(self, state, action, reward, next_state, done) -> None:
        experience = Experience(state, action, reward,  next_state, done)
        self.memory.append(experience)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Grab a random batch of samples
        samples = random.sample(self.memory, k=self.batch_size)
        
        # Grab S, A, R, S_, Done from the samples and store them into a vstack
        # where each row is just a 
        states = np.vstack([sample.state for sample in samples])
        actions = np.vstack([sample.action for sample in samples])
        rewards = np.vstack([sample.reward for sample in samples])
        next_states = np.vstack([sample.next_state for sample in samples])
        dones = np.vstack([sample.done for sample in samples])

        # Convert the above to tensors
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).byte().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)
