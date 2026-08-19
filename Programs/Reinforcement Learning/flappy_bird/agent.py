import flappy_bird_gymnasium
import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import argparse
import os
import warnings
import numpy as np

# Suppress observation space warnings from gymnasium
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# Set up GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent():
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

            self.alpha = params["alpha"]
            self.gamma = params["gamma"] 
            self.epsilon_init = params["epsilon_init"]
            self.epsilon_min = params["epsilon_min"]
            self.epsilon_decay = params["epsilon_decay"]
            self.replay_memory_size = params["replay_memory_size"]
            self.mini_batch_size = params["mini_batch_size"]
            self.network_sync_rate = params["network_sync_rate"]
            self.reward_threshold = params["reward_threshold"]

            self.loss_fn = nn.MSELoss()
            self.optimizer = None 
            self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
            self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Unpack the batch of raw CPU values
        states, actions, next_states, rewards, terminateds = zip(*mini_batch)

        # Convert to numpy arrays first (for speed), then to tensors, then move to GPU
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        terminateds = torch.tensor(terminateds, dtype=torch.float32, device=device)

        with torch.no_grad():
            target_q = rewards + (1 - terminateds) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self, is_training=True, render=False):
        # Number of parallel environments (adjust based on your CPU cores)
        num_envs = 8 

        if is_training:
            # "async" enables Multiprocessing across CPU cores
            env = gymnasium.make_vec("FlappyBird-v0", num_envs=num_envs, vectorization_mode="async")
        else:
            # Run single environment for evaluation
            env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)
            
        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n
        policy_dqn = DQN(num_state, num_action).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_state, num_action).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval()
            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
            
            # Vector environments return arrays of states
            states, _ = env.reset()
            episode_rewards = np.zeros(num_envs)
            global_episodes = 0
            
            print(f"Training started with {num_envs} parallel environments...")

            # Vector envs auto-reset, so we use an infinite loop and break on conditions
            while True:
                # 1. Epsilon-Greedy Action Selection for multiple environments
                random_actions = env.action_space.sample() # Returns array of 8 random actions
                
                with torch.no_grad():
                    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                    q_values = policy_dqn(states_tensor)
                    model_actions = q_values.argmax(dim=1).cpu().numpy()
                
                # Apply epsilon greedily across all parallel environments
                mask = np.random.rand(num_envs) < epsilon
                actions = np.where(mask, random_actions, model_actions)
                
                # 2. Step all environments simultaneously
                next_states, rewards, terminateds, truncateds, infos = env.step(actions)
                
                # 3. Process each environment's transition
                for i in range(num_envs):
                    episode_rewards[i] += rewards[i]
                    done = terminateds[i] or truncateds[i]
                    
                    # Because VectorEnvs auto-reset, we must extract the TRUE final observation
                    if done and "_final_observation" in infos and infos["_final_observation"][i]:
                        actual_next_state = infos["final_observation"][i]
                    else:
                        actual_next_state = next_states[i]

                    # Save to memory (CPU)
                    memory.append((states[i], actions[i], actual_next_state, rewards[i], terminateds[i]))
                    
                    # Handle episode completion
                    if done:
                        global_episodes += 1
                        print(f"Episode: {global_episodes} | Reward: {episode_rewards[i]:.2f} | Epsilon {epsilon:.4f}")
                        
                        if episode_rewards[i] > best_reward:
                            best_reward = episode_rewards[i]
                            log_msg = f"Best Reward: {best_reward} for episode: {global_episodes}"
                            
                            with open(self.LOG_FILE, "a") as f:
                                f.write(log_msg + "\n")
                            torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                            
                        if episode_rewards[i] >= self.reward_threshold:
                            print(f"\nSolved! Reward threshold reached.")
                            env.close()
                            return
                        
                        # Reset reward counter for this specific environment
                        episode_rewards[i] = 0 
                        
                # 4. Global updates
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                steps += num_envs
                
                # 5. Optimize the network
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.random_sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

                states = next_states
                
        else:
            # Single-environment evaluation loop
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

            for episode in itertools.count():
                state, _ = env.reset()
                episode_reward = 0
                done = False

                while not done and episode_reward < self.reward_threshold:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = policy_dqn(state_tensor).argmax().item()
                    
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state

                print(f"Evaluation Episode: {episode} | Reward: {episode_reward:.2f}")
                if episode_reward >= self.reward_threshold:
                    break

        env.close()
    
if __name__ == "__main__":
    import multiprocessing
    
    # REQUIRED FOR WINDOWS MULTIPROCESSING: Prevents recursive process spawning
    multiprocessing.freeze_support() 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("hyperparameters", help='')
    parser.add_argument("--train", help='Training Mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)