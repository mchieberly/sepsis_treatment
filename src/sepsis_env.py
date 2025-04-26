import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SepsisEnv(gym.Env):
	def __init__(self):
		super(SepsisEnv, self).__init__()
		# State space: [heart_rate, blood_pressure, lactate_level]
		# Discretized for simplicity: 3 levels each (low, normal, high)
		self.observation_space = spaces.Box(
			low=np.array([0, 0, 0]), high=np.array([2, 2, 2]), dtype=np.int32
		)
		# Action space: 0: no action, 1: antibiotics, 2: fluids, 3: vasopressors
		self.action_space = spaces.Discrete(4)
		# Initial patient state
		self.state = None
		self.max_steps = 48  # 48 hours
		self.current_step = 0

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		# Random initial state (mimicking MIMIC-III patient)
		self.state = np.random.randint(0, 3, size=3)  # [heart_rate, blood_pressure, lactate]
		self.current_step = 0
		return self.state, {}

	def step(self, action):
		self.current_step += 1
		# Simplified state transition (replace with MIMIC-III data-driven logic)
		# Example: antibiotics (action=1) may reduce lactate if high
		new_state = self.state.copy()
		reward = 0

		if action == 1 and self.state[2] > 0:  # Antibiotics reduce lactate
			new_state[2] = max(0, self.state[2] - 1)
			reward += 2
		elif action == 2 and self.state[1] < 2:  # Fluids increase blood pressure
			new_state[1] = min(2, self.state[1] + 1)
			reward += 1
		elif action == 3 and self.state[1] < 2:  # Vasopressors increase blood pressure
			new_state[1] = min(2, self.state[1] + 1)
			reward += 1
		elif action == 0:  # No action
			pass
		else:
			reward -= 1  # Penalty for inappropriate action

		# Random deterioration (simulating sepsis progression)
		if np.random.random() < 0.2:
			new_state[np.random.randint(0, 3)] = min(2, new_state[np.random.randint(0, 3)] + 1)
			reward -= 2

		self.state = new_state

		# Reward: +10 if patient stabilizes (normal state), -10 if critical
		if np.all(self.state == 1):  # All normal
			reward += 10
		elif np.any(self.state == 2):  # Any critical
			reward -= 5

		# Termination: max steps or patient stabilizes/deteriorates
		terminated = self.current_step >= self.max_steps or np.all(self.state == 1)
		truncated = np.all(self.state == 2)  # All critical (simulating death)
		return self.state, reward, terminated, truncated, {}

	def render(self):
		print(f"Step: {self.current_step}, State: HR={self.state[0]}, BP={self.state[1]}, Lactate={self.state[2]}")

