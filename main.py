# AGENT

import gym
import torch
import os
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
gym.envs.register(
id='Godzilla-v0',
entry_point='my_env_module:Godzilla'
)

env = gym.make('Godzilla-v0')


# create the agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 32
n_episodes = 1000

output_dir = 'model/cartpole'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
done = False

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(5000):
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10

        next_state = np.reshape(next_state, [1,state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print('episode: {}/{}, score: {}, e: {:.2}'.format(e, n_episodes, time, agent.epsilon))
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + 'weights_' + '{:04d}'.format(e) + 'hdf5')


