import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DQN experiments for CartPole-v1")
    # Modes: naive, only_tn, only_er, tn_er
    parser.add_argument("--mode", type=str, choices=["naive", "only_tn", "only_er", "tn_er"], default="naive",
                        help="Experiment mode: naive (no TN, no ER), only_tn (TN only), only_er (ER only), tn_er (both TN and ER)")
    parser.add_argument("--max_steps", type=int, default=1000000,
                        help="Total environment steps to run (default 1e6).")
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (applies if using ER).")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon.")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum epsilon.")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--memory_size", type=int, default=10000, help="Replay buffer size (if ER is used).")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden layer size for the network.")

    parser.add_argument("--update_to_data", type=int, default=1,
                        help="For naive DQN: number of environment steps to collect before one gradient update.")
    return parser.parse_args()

args = parse_args()

USE_TARGET_NETWORK = (args.mode in ["only_tn", "tn_er"])
USE_REPLAY_BUFFER = (args.mode in ["only_er", "tn_er"])


# Neural Network for Q-function
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# class DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=args.memory_size) if USE_REPLAY_BUFFER else None
        self.model = DQN(state_size, action_size, hidden_size=args.hidden_size)
        if USE_TARGET_NETWORK:
            self.target_model = DQN(state_size, action_size, hidden_size=args.hidden_size)
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.target_model = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.epsilon = args.epsilon_start

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, transition):
        if self.memory is not None:
            self.memory.append(transition)

    def update_target_model(self):
        if self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        if self.memory is None or len(self.memory) < args.batch_size:
            return
        batch = random.sample(self.memory, args.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        if self.target_model is not None:
            next_q = self.target_model(next_states).max(1)[0].detach()
        else:
            next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * args.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_naive(self, transition):
        state, action, reward, next_state, done = transition
        state_tensor = torch.FloatTensor([state])
        next_state_tensor = torch.FloatTensor([next_state])
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([done])

        current_q = self.model(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze()
        if self.target_model is not None:
            next_q = self.target_model(next_state_tensor).max(1)[0].detach()
        else:
            next_q = self.model(next_state_tensor).max(1)[0].detach()

        target_q = reward_tensor + (1 - done_tensor) * args.gamma * next_q
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

total_steps = 0
episode_count = 0
scores = []
env_steps_list = []

naive_buffer = []

while total_steps < args.max_steps:
    episode_count += 1
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    episode_reward = 0

    while not done and total_steps < args.max_steps:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated
        episode_reward += reward
        total_steps += 1

        transition = (state, action, reward, next_state, float(done))
        
        if USE_REPLAY_BUFFER:
            agent.remember(transition)
            agent.replay()
        else:
            naive_buffer.append(transition)
            if len(naive_buffer) >= args.update_to_data:
                for trans in naive_buffer:
                    agent.update_naive(trans)
                naive_buffer = []
        
        state = next_state

    scores.append(episode_reward)
    env_steps_list.append(total_steps)

    if USE_TARGET_NETWORK and (episode_count % 5 == 0):
        agent.update_target_model()

    agent.epsilon = max(args.epsilon_end, agent.epsilon * args.epsilon_decay)
    print(f"Episode: {episode_count}, Score: {episode_reward}, Total Steps: {total_steps}, Epsilon: {agent.epsilon:.3f}")

# for plots
results = pd.DataFrame({
    'Episode_Return': scores,
    'env_step': env_steps_list
})

results['Episode_Return_smooth'] = results['Episode_Return'].rolling(window=10, min_periods=1).mean()

output_filename = f"results/naive_with_lr_ab/0.001/data_lr_0.001_4.csv"
results.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")

plt.plot(results['env_step'], results['Episode_Return'], label='Episode Return')
plt.plot(results['env_step'], results['Episode_Return_smooth'], label='Smoothed Return')
plt.xlabel('Environment Steps')
plt.ylabel('Return')
plt.title(f"DQN Training (Mode: {args.mode}), LR = 0.001")
plt.legend()
plt.show()
