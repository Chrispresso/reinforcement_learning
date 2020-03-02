import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

import random
from collections import deque, namedtuple
from typing import List, Dict, Any, Optional, Tuple

from .experience_replay import ExperienceReplayBuffer, Experience

params: Dict[str, Any] = {
    'replay_buffer_size': 10000,
    'batch_size': 32,
    'discount': .99,
    'optimizer': optim.Adam,
    'optimizer_kwargs': {},
    'loss_func': F.mse_loss,
    'hidden_layers': [32, 24],
    'exploration_type': 'e-greedy',
    'tau': 0.9,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'eps_decay': 0.98,
}

device = 'cpu'


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
        x = state
        for i in range(len(self.linears) - 1):
            x = F.relu(self.linears[i](x))
        x = self.linears[-1](x)
        return x


class DQNAgent():
    def __init__(self, num_states: int, num_actions: int, params):
        self.num_states = num_states
        self.num_actions = num_actions
        self.params = params

        self.policy_network = QNetwork(self.num_states, self.num_actions, self.params['hidden_layers'])
        
        self.optimizer = self.params['optimizer'](self.policy_network.parameters(), **self.params['optimizer_kwargs'])

        self.memory = ExperienceReplayBuffer(self.params['replay_buffer_size'], self.params['batch_size'])

        self.time_step = 1

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        # states, actions, rewards, next_states and dones are all shape [batch_size, 1]
        states, actions, rewards, next_states, dones = experiences

        # Get state-action values for states (shape [batch, 1])
        state_action_vals = self.policy_network(states).gather(1, actions)

        # Get state-action values for next_states assuming greedy-policy
        # unsqueeze to go from shape [batch] to [batch, 1]
        state_action_vals_next_states = self.policy_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute expected
        expected_state_action_values = rewards + (self.params['discount'] * state_action_vals_next_states * (1 - dones))
        
        # Clear gradient and minimize
        self.policy_network.train()
        self.optimizer.zero_grad()
        loss = self.params['loss_func'](state_action_vals, expected_state_action_values)
        loss.backward()
        self.optimizer.step()

        # self.soft_update()

    def soft_update(self):
        tau = self.params['tau']
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.add_experience(state, action, reward, next_state, done)

        if len(self.memory) > self.params['batch_size']:
            experiences = self.memory.sample()
            self.learn(experiences)
        
        self.time_step += 1

    def act(self, state: np.ndarray, eps) -> int:
        # Convert state to [1, N] where N is the number of state dimensions
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.policy_network.eval()
        with torch.no_grad():
            action_vals = self.policy_network(state)
        self.policy_network.train()

        if random.random() < eps:
            return random.choice(np.arange(self.num_actions))
        else:
            return np.argmax(action_vals.cpu().data.numpy())
