import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Neural Network
class QNetwork(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(QNetwork, self).__init__()
		self.fc1 = nn.Linear(state_dim, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, action_dim)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
		return (np.array(state), np.array(action), np.array(reward),
				np.array(next_state), np.array(done))

	def __len__(self):
		return len(self.buffer)

# DDQN Agent
class DDQNAgent:
	def __init__(self, state_dim, action_dim, device):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.loss_fn = nn.MSELoss()
		self.policy_net = QNetwork(state_dim, action_dim).to(device)
		self.target_net = QNetwork(state_dim, action_dim).to(device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
		self.memory = ReplayBuffer(10000)
		self.batch_size = 64
		self.gamma = 0.99
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.tau = 0.005

	def act(self, state):
		if random.random() < self.epsilon:
			return random.randrange(self.action_dim)
		state = torch.FloatTensor(state).to(self.device)
		with torch.no_grad():
			q_values = self.policy_net(state)
		return q_values.argmax().item()

	def update(self):
		if len(self.memory) < self.batch_size:
			return
		states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
		states = torch.FloatTensor(states).to(self.device)
		actions = torch.LongTensor(actions).to(self.device)
		rewards = torch.FloatTensor(rewards).to(self.device)
		next_states = torch.FloatTensor(next_states).to(self.device)
		dones = torch.FloatTensor(dones).to(self.device)

		# Compute Q-values
		q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
		
		# Use policy net to select actions and target net to evaluate
		next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
		next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
		targets = rewards + (1 - dones) * self.gamma * next_q_values

		# Compute loss
		loss = self.loss_fn(q_values, targets.detach())

		# Optimize
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Soft update target network
		for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
			target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

		# Decay epsilon
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

