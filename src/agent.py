import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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

class DDQNAgent:
	def __init__(self, state_dim, action_dim, device):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device

		# Hyperparameters
		self.gamma = 0.995  			# Discount factor
		self.epsilon = 1.0  			# Exploration rate
		self.epsilon_min = 0.1  		# Minimum exploration rate
		self.epsilon_decay = 0.995  	# Exploration decay rate
		self.memory_size = 50000  		# Replay buffer size
		self.batch_size = 128  			# Batch size for training
		self.tau = 0.01  				# Soft update parameter

		# Networks
		self.policy_net = QNetwork(state_dim, action_dim).to(device)
		self.target_net = QNetwork(state_dim, action_dim).to(device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()

		# Optimizer
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

		# Replay buffer
		self.memory = deque(maxlen=self.memory_size)

	def act(self, state):
		if random.random() < self.epsilon:
			return random.randrange(self.action_dim)
		state = torch.FloatTensor(state).to(self.device)
		with torch.no_grad():
			q_values = self.policy_net(state)
		return q_values.argmax().item()

	def decay_epsilon(self):
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

	def push(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def update(self):
		if len(self.memory) < self.batch_size:
			return 0.0

		batch = random.sample(self.memory, self.batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)

		# Convert lists to PyTorch tensors
		states = torch.FloatTensor(np.array(states)).to(self.device)
		actions = torch.LongTensor(np.array(actions)).to(self.device)
		rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
		next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
		dones = torch.FloatTensor(np.array(dones)).to(self.device)

		# Current Q-values
		q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

		# Use policy net to select actions, target net to evaluate
		next_actions = self.policy_net(next_states).argmax(1)
		next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
		targets = rewards + (1 - dones) * self.gamma * next_q_values

		# Compute loss
		loss = nn.MSELoss()(q_values, targets.detach())

		# Optimize
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Soft update target network
		for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
			target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

		return loss.item()
