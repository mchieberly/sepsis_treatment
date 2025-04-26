import gymnasium as gym
import torch
from src.sepsis_env import SepsisEnv
from src.model import DDQNAgent

def train_ddqn():
	env = SepsisEnv()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	agent = DDQNAgent(state_dim=3, action_dim=4, device=device)
	episodes = 1000
	total_rewards = []

	for episode in range(episodes):
		state, _ = env.reset()
		episode_reward = 0
		done = False
		while not done:
			action = agent.act(state)
			next_state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			agent.memory.push(state, action, reward, next_state, done)
			agent.update()
			state = next_state
			episode_reward += reward
			if done:
				break
		total_rewards.append(episode_reward)
		if episode % 10 == 0:
			print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

	# Evaluate the policy
	env = SepsisEnv()
	state, _ = env.reset()
	done = False
	while not done:
		action = agent.act(state)
		state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated
		env.render()

if __name__ == "__main__":
	train_ddqn()

