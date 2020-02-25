import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

import random
from collections import deque, namedtuple
from typing import List, Dict, Any, Optional, Tuple

params: Dict[str, Any] = {
    'replay_buffer_size': 10000,
    'batch_size': 32,
    'optimizer': optim.Adam,
    'optimizer_kwargs': {},
    'exploration_type': 'softmax',
    'tau': 0.9,
    'eps_start': 1.0,
    'eps_end': 0.02,
    'eps_decay': 0.98,
    # 'test_multiple_params': True
}

device = 'cpu'
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    def __init__(self, num_states: int, num_actions: int, hidden_layers: List[int]):
        """
        num_states: Number of states
        num_actions: Number of actions. Cannot be used in a continuous action space
        hidden_layers: A list of integers containing the number of hidden units in each layer
        """
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_layers = hidden_layers
        self.linears = nn.ModuleList()

        # Add (states, hidden_layer[0])
        self.linears.append(nn.Linear(self.num_states, self.hidden_layers[0]))

        # Add (hidden_layer[i], hidden_layer[i+1])
        for i in range(0, len(self.hidden_layers) - 1):
            in_features = self.hidden_layers[i]
            out_features = self.hidden_layers[i + 1]
            self.linears.append(nn.Linear(in_features, out_features))

        # Add (hidden_layers[-1], actions)
        self.linears.append(nn.Linear(self.hidden_layers[-1], self.num_actions))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state.float()
        for linear in self.linears:
            x = F.relu(linear(x))
        return x


class ExperienceReplayBuffer():
    def __init__(self, buffer_size: int, batch_size: int):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add_experience(self, state, action, reward, next_state, done) -> None:
        experience = Experience(state, action, reward,  next_state, done)
        self.memory.append(experience)

    def sample(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Only sample if we have enough to take a sample
        if len(self.memory) < self.batch_size:
            return

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
        actions = torch.from_numpy(actions).int().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).byte().to(device)

        return states, actions, rewards, next_states, dones

class DQNAgent():
    def __init__(self, num_states: int, num_actions: int, params=params):
        self.num_states = num_states
        self.num_actions = num_actions
        self.params = params

        # Create Q_target and Q (which is just a copy of Q_target for now)
        self.Q_target = QNetwork(self.num_states, self.num_actions)
        self.Q = QNetwork(self.num_states, self.num_actions)
        self.Q.load_state_dict(self.Q_target.state_dict())

        # Create optimizer
        self.optimizer = self.params['optimizer'](self.Q.parameters(), **self.params['optimizer_params'])

        self.memory = ExperienceReplayBuffer(self.params['replay_buffer_size'], self.params['batch_size'])

    def get_action(self, state: np.ndarray) -> int:
        # Get action-values from Q(S, :)
        state = torch.from_numpy(state).float().to(device)
        self.Q.eval()
        with torch.no_grad():
            action_values = self.Q(state)
        # Re-enable training
        self.Q.train()

        #@TODO: Do action selection

    def _epsilon_greedy_action_selection(self, action_values: torch.FloatStorage) -> int:
        # At a rate of epsilon, select a uniform random action
        if random.random() < self.params['eps'])
            action = np.random.choice(np.arange(self.num_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())
        
        self.params['eps'] = max(self.params['eps'] * self.params['eps_decay'], self.params['eps_end'])
        return action

    def _softmax_action_selection(self, action_values: torch.FloatStorage) -> int:
        preferences = action_values.cpu().data.numpy() / self.params['tau']