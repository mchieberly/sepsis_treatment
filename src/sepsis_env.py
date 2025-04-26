import gymnasium as gym
import numpy as np
from gymnasium import spaces
import json

class SepsisEnv(gym.Env):
	def __init__(self, transition_probs_file='processed_data/transition_probs.json'):
		super(SepsisEnv, self).__init__()
		self.observation_space = spaces.Box(
			low=np.array([0, 0, 0]), high=np.array([2, 2, 2]), dtype=np.int32
		)
		self.action_space = spaces.Discrete(4)
		self.state = None
		self.max_steps = 48
		self.current_step = 0

		# Load transition probabilities
		with open(transition_probs_file, 'r') as f:
			self.transition_probs = json.load(f)
			self.transition_probs = {
				eval(k): {eval(kk): vv for kk, vv in v.items()}
				for k, v in self.transition_probs.items()
			}

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.state = np.random.randint(0, 3, size=3)
		self.current_step = 0
		return self.state, {}

	def step(self, action):
		self.current_step += 1
		current_state = tuple(self.state)
		key = (current_state, action)

		# Get next state probabilities
		probs = self.transition_probs.get(key, None)
		if probs:
			next_states = list(probs.keys())
			next_probs = list(probs.values())
			next_state = next_states[np.random.choice(len(next_states), p=next_probs)]
		else:
			# Fallback to random state if no data
			next_state = np.random.randint(0, 3, size=3)

		self.state = np.array(next_state)

		# Compute reward
		reward = 0
		if action == 1 and current_state[2] > 0:	# Antibiotics
			reward += 2
		elif action == 2 and current_state[1] < 2:	# Fluids
			reward += 1
		elif action == 3 and current_state[1] < 2:	# Vasopressors
			reward += 1
		elif action == 0:
			pass
		else:
			reward -= 1

		if np.all(self.state == 1):		# All normal
			reward += 10
		elif np.any(self.state == 2):	# Any critical
			reward -= 5

		terminated = self.current_step >= self.max_steps or np.all(self.state == 1)
		truncated = np.all(self.state == 2)
		return self.state, reward, terminated, truncated, {}

	def render(self):
		print(f"Step: {self.current_step}, State: HR={self.state[0]}, BP={self.state[1]}, Lactate={self.state[2]}")
