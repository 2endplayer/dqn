import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from gym.envs.registration import register

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# AGENT

import torch
import torch.nn as nn
import torch.optim as optim
import random

from collections import deque
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        return model, optimizer, criterion

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.from_numpy(state).float()
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state_tensor)
        self.model.train()
        return np.argmax(action_values.numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.from_numpy(state).float()
            next_state_tensor = torch.from_numpy(next_state).float()

            with torch.no_grad():
                target = self.model(state_tensor)
                target_next = self.model(next_state_tensor)

            target[action] = reward
            if not done:
                target[action] += self.gamma * torch.max(target_next)

            states.append(state)
            targets.append(target)

        states_tensor = torch.from_numpy(np.vstack(states)).float()
        targets_tensor = torch.stack(targets)

        self.model.train()
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(states_tensor), targets_tensor)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


# ENVIRONMENT
from gym_examples.my_env_module import Godzilla

env = gym.make('my_env_module.py:Godzilla')

# create the agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
