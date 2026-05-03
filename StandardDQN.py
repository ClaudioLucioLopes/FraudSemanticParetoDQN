import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from FraudDataHandler import FraudDataHandler
from FraudDataHandlerImproved import FraudDataHandlerImproved
from FraudMOEnv import FraudMOEnv

# Technical Detail: Q-Network Architecture
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, state):
        return self.net(state)

# Technical Detail: Standard Replay Buffer
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class StandardDQNAgent:
    def __init__(self, state_dim, action_dim=2, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        # Improved device selection with compatibility check
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                self.device = torch.device("cuda")
            else:
                print(f"Warning: GPU {torch.cuda.get_device_name()} (CC {major}.{minor}) "
                      f"is incompatible with this PyTorch build. Falling back to CPU in StandardDQNAgent.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayMemory(capacity=50000)
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_dim = action_dim

    def select_action(self, state, inference=False):
        """Epsilon-greedy action selection."""
        if not inference and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self, batch_size=64):
        """Standard TD-Error optimization."""
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        curr_Q = self.q_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            max_next_Q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_Q = rewards + (1 - dones) * self.gamma * max_next_Q
            
        loss = nn.functional.mse_loss(curr_Q, target_Q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

