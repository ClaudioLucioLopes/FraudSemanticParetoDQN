import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymoo.indicators.hv import HV
from collections import deque
import random
import copy

from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# 1. Vectorial Replay Memory (Restored for MDP Probability Integrity)
# ---------------------------------------------------------------------------
class VectorReplayMemory:
    def __init__(self, capacity=150000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward_vec, next_state, done):
        self.buffer.append((state, action, reward_vec, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------------------------
# 2. Neural Network Approximators 
# ---------------------------------------------------------------------------
class RewardApproximator(nn.Module):
    """Predicts immediate vectorial reward R(s, a) in R^3"""
    def __init__(self, state_dim, action_dim=1, nO=3):
        super(RewardApproximator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, nO)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class NonDominatedApproximator(nn.Module):
    """
    Approximates the Pareto frontier surface.
    Input: state + (d-1) sampled objectives + action
    Output: the d-th objective value
    """
    def __init__(self, state_dim, action_dim=1, nO=3):
        super(NonDominatedApproximator, self).__init__()
        self.nO = nO
        # Input: state + (2 objective sample points) + 1 action dimension
        input_dim = state_dim + (self.nO - 1) + action_dim
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, state, point, action):
        x = torch.cat([state, point, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

# ---------------------------------------------------------------------------
# 3. Hypervolume-Based Pareto Agent
# ---------------------------------------------------------------------------
class ParetoFraudAgent:
    def __init__(self, env_bounds, state_dim=384, num_objectives=3, gamma=0.98, epsilon=1.0, device='cpu'):
        self.nO = num_objectives
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995 # Decay per step
        self.device = device

        # Current Networks
        self.rew_net = RewardApproximator(state_dim, action_dim=1, nO=self.nO).to(device)
        self.nd_net = NonDominatedApproximator(state_dim, action_dim=1, nO=self.nO).to(device)
        
        # Target Networks (Critical for Stability)
        self.rew_target = copy.deepcopy(self.rew_net).to(device)
        self.nd_target = copy.deepcopy(self.nd_net).to(device)
        
        self.rew_opt = torch.optim.Adam(self.rew_net.parameters(), lr=1e-4)
        self.nd_opt = torch.optim.Adam(self.nd_net.parameters(), lr=1e-3)
        
        self.memory = VectorReplayMemory(capacity=150000)
        self.env_bounds = np.array([
            env_bounds['eff'],
            env_bounds['drift'],
            env_bounds['div']
        ])
        

        self.ref_point = np.zeros(self.nO)

    def sample_objective_points(self, n_samples=10):
        """Samples the (d-1) objective plane using infinite-horizon Q-value bounds."""
        q_scale = 1.0 / (1.0 - self.gamma)
        
        rand_points = torch.rand((n_samples, self.nO - 1), device=self.device)
        mins = torch.FloatTensor(self.env_bounds[:2, 0] * q_scale).to(self.device)
        maxs = torch.FloatTensor(self.env_bounds[:2, 1] * q_scale).to(self.device)
        ranges = maxs - mins
        
        scaled_points = (rand_points * ranges) + mins
        return scaled_points

    def update_target_networks(self):
        self.rew_target.load_state_dict(self.rew_net.state_dict())
        self.nd_target.load_state_dict(self.nd_net.state_dict())

    def evaluate_actions(self, state, n_samples=50):
        """Single-state evaluation for inference."""
        self.rew_net.eval()
        self.nd_net.eval()
        
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0) 
        actions_t = torch.FloatTensor([[0.0], [1.0]]).to(self.device)   
        
        with torch.no_grad():
            s_expanded = state_t.repeat(2, 1)
            r_pred = self.rew_net(s_expanded, actions_t) 
            
            points = self.sample_objective_points(n_samples) 
            s_tiled = state_t.repeat(2 * n_samples, 1)
            a_tiled = actions_t.repeat_interleave(n_samples, dim=0)
            p_tiled = points.repeat(2, 1)
            
            nd_pred_last = self.nd_net(s_tiled, p_tiled, a_tiled)
            nd_full = torch.cat([p_tiled, nd_pred_last], dim=1)
            nd_sets = nd_full.view(2, n_samples, self.nO)
            
            r_expanded = r_pred.unsqueeze(1) 
            q_sets = r_expanded + self.gamma * nd_sets
            
        self.rew_net.train()
        self.nd_net.train()
        return q_sets.cpu().numpy()

    def evaluate_actions_batch(self, states_batch, n_samples=10, use_target=False):
        """
        Batched forward pass evaluating all candidates (Actions 0 and 1) for a batch of states.
        SCIENTIFIC FIX: Decoupled logic for Double DQN.
        """
        self.rew_net.eval()
        self.nd_net.eval()
        
        batch_size = states_batch.size(0)
        actions_0 = torch.zeros(batch_size, 1).to(self.device)
        actions_1 = torch.ones(batch_size, 1).to(self.device)
        
        rew_network = self.rew_target if use_target else self.rew_net
        nd_network = self.nd_target if use_target else self.nd_net
        
        with torch.no_grad():
            # 1. Estimate Immediate Vectorial Rewards
            r_0 = rew_network(states_batch, actions_0) 
            r_1 = rew_network(states_batch, actions_1) 
            
            # 2. Estimate Non-Dominated Future Returns
            points = self.sample_objective_points(n_samples) 
            
            s_tiled = states_batch.repeat_interleave(n_samples, dim=0)
            a0_tiled = actions_0.repeat_interleave(n_samples, dim=0)
            a1_tiled = actions_1.repeat_interleave(n_samples, dim=0)
            p_tiled = points.repeat(batch_size, 1)
            
            # Predict the 3rd objective 
            nd_0 = nd_network(s_tiled, p_tiled, a0_tiled)
            nd_1 = nd_network(s_tiled, p_tiled, a1_tiled)
            
            nd_full_0 = torch.cat([p_tiled, nd_0], dim=1).view(batch_size, n_samples, self.nO)
            nd_full_1 = torch.cat([p_tiled, nd_1], dim=1).view(batch_size, n_samples, self.nO)
            
            # 3. Calculate Final Q_set
            q_sets_0 = r_0.unsqueeze(1) + self.gamma * nd_full_0
            q_sets_1 = r_1.unsqueeze(1) + self.gamma * nd_full_1
            
        self.rew_net.train()
        self.nd_net.train()
        
        return torch.stack([q_sets_0, q_sets_1], dim=1).cpu().numpy()

    def compute_hypervolumes(self, q_sets):
        """Computes hypervolume with independent axis normalization to prevent dimension dominance."""
        hvs = np.zeros(2)
        ref_point_offset = np.ones(self.nO) * -1 
        ind = HV(ref_point=ref_point_offset) 
        
        # Combine to find empirical bounds PER OBJECTIVE (axis=0)
        combined_q = np.concatenate([q_sets[0], q_sets[1]], axis=0)
        mins = np.min(combined_q, axis=0)
        maxs = np.max(combined_q, axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-5] = 1e-5 # Prevent division by zero
        
        for a in range(2):
            norm_points = (q_sets[a] - mins) / ranges
            
            # Pymoo expects a minimization problem, negate to [-1, 0]
            neg_points = norm_points * -1.0
            try:
                hvs[a] = ind(neg_points)
            except Exception:
                hvs[a] = 0.0 
        return hvs


    # def select_action(self, state, inference=False, n_samples=50):
    #     """Action selection strictly maximizing the raw topological Hypervolume."""
    #     if not inference and np.random.rand() < self.epsilon:
    #         return np.random.randint(2)
        
    #     # 1. Evaluate the continuous Pareto surface for both actions
    #     q_sets = self.evaluate_actions(state, n_samples=n_samples)
        
    #     # 2. Compute exact normalized Hypervolumes
    #     hvs = self.compute_hypervolumes(q_sets)
        
    #     # 3. Pure Pareto Inference: Pick the action with the largest volume
    #     best_indices = np.argwhere(hvs == np.amax(hvs)).flatten()
    #     return int(np.random.choice(best_indices))

    def select_action(self, state, inference=False, n_samples=50, return_prob=False):
        """
        Action selection. 
        Training: Explores raw topology.
        Inference: Uses Compromise Programming (Tchebycheff Distance to Utopia Point).
        """
        if not inference and np.random.rand() < self.epsilon:
            if return_prob: return np.random.randint(2), 0.5
            return np.random.randint(2)
        
        # q_sets shape: (2 actions, n_samples, n_objectives)
        q_sets = self.evaluate_actions(state, n_samples=n_samples)
        
        if not inference:
            # Training: Standard Hypervolume maximization for policy mapping
            hvs = self.compute_hypervolumes(q_sets)
            best_indices = np.argwhere(hvs == np.amax(hvs)).flatten()
            return int(np.random.choice(best_indices))
            
        else:
            # Automated Compromise Programming for Inference
            # 1. Collapse the distribution to expected vectorial values
            q_expected = np.mean(q_sets, axis=1) # Shape: (2, 3)
            
            # 2. Dynamically normalize the objective space to [0, 1]
            mins = np.min(q_expected, axis=0)
            maxs = np.max(q_expected, axis=0)
            ranges = maxs - mins
            ranges[ranges < 1e-5] = 1e-5 # Prevent division by zero
            
            q_norm = (q_expected - mins) / ranges
            
            # 3. Define the local Utopia Point (Maximum of each normalized objective = 1.0)
            utopia_point = np.ones(self.nO)
            
            # 4. Calculate Augmented Tchebycheff Distance to Utopia
            # Tchebycheff strictly minimizes the worst-case objective degradation
            distances = np.zeros(2)
            augmentation_factor = 0.01 # Small factor to prevent weakly Pareto optimal selection
            
            for a in range(2):
                diff = np.abs(utopia_point - q_norm[a])
                max_diff = np.max(diff)
                sum_diff = np.sum(diff)
                # L_infinity norm + small L_1 norm for strict dominance
                distances[a] = max_diff + (augmentation_factor * sum_diff)
                
            # 5. Select the action that is mathematically closest to the Utopia point
            action = int(np.argmin(distances))
            
            if return_prob:
                raw_q_0 = np.sum(q_expected[0]) # Raw predicted utility of Action 0
                raw_q_1 = np.sum(q_expected[1]) # Raw predicted utility of Action 1
                
                # Derive temperature natively from the network's own sample variance
                temperature = np.std(np.sum(q_sets, axis=2)) + 1e-3
                q_diff = raw_q_1 - raw_q_0
                
                # Logistic Sigmoid maps the unbounded continuous Q-values exactly to [0.0, 1.0]
                prob = 1.0 / (1.0 + np.exp(-q_diff / temperature))
                return action, prob
            
            return action


    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self, batch_size=128):
        """
        Optimizes Approximators using Double Pareto-DQN Logic.
        """
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        actual_batch_size = states.shape[0]
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device) 
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # --- 1. Train Reward Approximator ---
        pred_rewards = self.rew_net(states, actions)

        # Apply right before loss calculation
        rewards_sym = symlog(rewards)
        pred_rewards_sym = symlog(pred_rewards) # Optional: predict raw, scale loss
        rew_loss = nn.functional.smooth_l1_loss(pred_rewards_sym, rewards_sym)
        # rew_loss = nn.functional.mse_loss(pred_rewards, rewards)
        
        self.rew_opt.zero_grad()
        rew_loss.backward()
        self.rew_opt.step()
        
        # --- 2. Train Non-Dominated Approximator (Double Topological Bellman) ---
        with torch.no_grad():
            # Use CURRENT network to select the action maximizing the topology
            q_sets_batch_current = self.evaluate_actions_batch(next_states, n_samples=10, use_target=False)
            
            best_next_actions = []
            for i in range(actual_batch_size):
                hvs = self.compute_hypervolumes(q_sets_batch_current[i])
                best_next_actions.append([float(np.argmax(hvs))])
                
            best_next_actions = torch.FloatTensor(best_next_actions).to(self.device)
            
            # Use TARGET network to evaluate the true value of that action
            sampled_points = self.sample_objective_points(n_samples=1)
            p_tiled = sampled_points.repeat(actual_batch_size, 1)
            
            next_r_best = self.rew_target(next_states, best_next_actions)
            next_nd_best = self.nd_target(next_states, p_tiled, best_next_actions)
            
            target_nd = next_r_best[:, -1].unsqueeze(1) + self.gamma * next_nd_best * (1 - dones)
            
        curr_nd = self.nd_net(states, p_tiled, actions)
        nd_loss = nn.functional.smooth_l1_loss(curr_nd, target_nd)
        
        self.nd_opt.zero_grad()
        nd_loss.backward()
        self.nd_opt.step()
        
        return nd_loss.item()