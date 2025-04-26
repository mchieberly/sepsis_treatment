import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import logging
from src.sepsis_env import SepsisEnv
from src.agent import DDQNAgent

# Set up directories and logging
os.makedirs('trained_model/progress', exist_ok=True)
logging.basicConfig(filename='trained_model/progress/training_log.txt', level=logging.INFO, 
					format='%(asctime)s - %(levelname)s - %(message)s')

# Training parameters
NUM_EPISODES = 10000
MIN_EPISODES = 2000
PATIENCE = 1000
BEST_REWARD_THRESHOLD = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(show_live_plot=False):
	env = SepsisEnv()
	agent = DDQNAgent(state_dim=3, action_dim=4, device=device)

	episode_rewards = []
	episode_losses = []
	avg_rewards = []
	avg_losses = []
	best_avg_reward = -float('inf')
	patience_counter = 0

	# Set up plot
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
	
	reward_line, = ax1.plot([], [], label='Episode Reward', alpha=0.5)
	avg_reward_line, = ax1.plot([], [], label='Avg Reward (100 episodes)', color='red')
	ax1.set_xlabel('Episode')
	ax1.set_ylabel('Reward')
	ax1.set_title('Training Rewards')
	ax1.legend()
	ax1.grid(True)

	loss_line, = ax2.plot([], [], label='Episode Loss', alpha=0.5)
	avg_loss_line, = ax2.plot([], [], label='Avg Loss (100 episodes)', color='red')
	ax2.set_xlabel('Episode')
	ax2.set_ylabel('Loss')
	ax2.set_title('Training Loss')
	ax2.legend()
	ax2.grid(True)

	# Add plot title
	plt.suptitle('DQN Training Progress for Sepsis Environment', fontsize=16)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)

	# Enable interactive mode and set window title if show_live_plot is True
	if show_live_plot:
		plt.ion()
		fig.canvas.manager.set_window_title('DQN Training Progress for Sepsis Environment')

	try:
		for episode in range(NUM_EPISODES):
			state, _ = env.reset()
			total_reward = 0
			total_loss = 0
			steps = 0
			
			while True:
				action = agent.act(state)
				next_state, reward, terminated, truncated, _ = env.step(action)
				done = terminated or truncated
				
				# Clip rewards to stabilize training
				clipped_reward = np.clip(reward, -1, 1)
				agent.push(state, action, clipped_reward, next_state, done)
				state = next_state
				total_reward += reward
				steps += 1
				
				loss = agent.update()
				total_loss += loss
				
				if done:
					agent.decay_epsilon()
					break
			
			# Log metrics
			avg_loss = total_loss / max(steps, 1)
			episode_rewards.append(total_reward)
			episode_losses.append(avg_loss)
			avg_reward = np.mean(episode_rewards[-100:]) if episode >= 100 else np.mean(episode_rewards)
			avg_rewards.append(avg_reward)
			avg_loss_val = np.mean(episode_losses[-100:]) if episode >= 100 else np.mean(episode_losses)
			avg_losses.append(avg_loss_val)
			
			logging.info(f"Episode {episode}, Reward: {total_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}, "
						 f"Epsilon: {agent.epsilon:.3f}, Loss: {avg_loss:.4f}, Steps: {steps}")
			
			if episode % 10 == 0:
				print(f"Episode {episode}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, "
					  f"Epsilon: {agent.epsilon:.3f}, Loss: {avg_loss:.4f}, Steps: {steps}")
			
			# Update plot data
			reward_line.set_xdata(range(len(episode_rewards)))
			reward_line.set_ydata(episode_rewards)
			avg_reward_line.set_xdata(range(len(avg_rewards)))
			avg_reward_line.set_ydata(avg_rewards)
			ax1.relim()
			ax1.autoscale_view()

			loss_line.set_xdata(range(len(episode_losses)))
			loss_line.set_ydata(episode_losses)
			avg_loss_line.set_xdata(range(len(avg_losses)))
			avg_loss_line.set_ydata(avg_losses)
			ax2.relim()
			ax2.autoscale_view()

			# Update display only if show_live_plot is True
			if show_live_plot:
				plt.draw()
				fig.canvas.flush_events()
				plt.pause(0.001)

			# Save model if improved
			if avg_reward > best_avg_reward:
				best_avg_reward = avg_reward
				torch.save(agent.policy_net.state_dict(), 'trained_model/best_dqn_sepsis.pth')
				patience_counter = 0
			else:
				patience_counter += 1
			
			# Early stopping
			if episode >= MIN_EPISODES and (avg_reward >= BEST_REWARD_THRESHOLD or patience_counter >= PATIENCE):
				logging.info(f"Stopping early at episode {episode}. Avg Reward: {avg_reward:.2f}")
				break
	
	except KeyboardInterrupt:
		print("\nTraining interrupted by user (Ctrl+C). Saving current model and plot...")
		logging.info(f"Training interrupted by user at episode {episode}. Avg Reward: {avg_reward:.2f}")
		
		# Turn off interactive mode if it was enabled
		if show_live_plot:
			plt.ioff()
		
		# Save the final plot
		plt.savefig('trained_model/progress/training_progress.png')
		plt.close()
		
		print("Model saved as 'trained_model/interrupted_dqn_sepsis.pth'. Plot saved as 'trained_model/progress/training_progress.png'.")
		return

	# Turn off interactive mode if it was enabled
	if show_live_plot:
		plt.ioff()
	
	# Save the final plot
	plt.savefig('trained_model/progress/training_progress.png')
	plt.close()

if __name__ == "__main__":
	  # Set show_live_plot=True to see the live plot during training
	train(show_live_plot=True)
