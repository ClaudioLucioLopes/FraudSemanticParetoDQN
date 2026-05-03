import numpy as np
import math
from collections import deque

class CreditMOEnv:
    def __init__(self, data, mode="pareto", W=1000, alpha=0.1, interest_rate=0.005, calib=None):
        """
        Multi-Objective Environment for Credit Default Risk.
        State: s_t = [Longitudinal_Embedding, Current_FPR]
        Action: 0 (Approve Credit), 1 (Reject Credit)
        """
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        self.raw_limits = data['limits'] # Adjusted from 'values'
        self.limits = np.log1p(self.raw_limits)
        
        self.mode = mode
        self.n_samples = len(self.labels)
        self.emb_dim = self.embeddings.shape[1] + 1 if mode == "pareto" else self.embeddings.shape[1]
        
        self.W = W
        self.alpha = alpha
        self.interest_rate = interest_rate 
        self.epsilon = 1e-5 

        # --- Calibration (Focusing on Limit Exposure) ---
        if calib is not None:
            self.c = calib['c']
            self.gamma = calib['gamma']
            self.div_scale = calib['div_scale']
            self.env_bounds = calib.get('env_bounds', None)
        else:
            legit_limits = self.limits[self.labels == 0]
            fraud_limits = self.limits[self.labels == 1]
            
            self.c = np.median(legit_limits) * 0.2 if len(legit_limits) > 0 else 2.5
                
            if len(fraud_limits) > 0:
                max_v = np.percentile(fraud_limits, 95)
                avg_rho = max(len(fraud_limits) / len(self.labels), 0.001)
                max_r_eff_penalty = max_v * ( (1.0 / avg_rho) * 25.0 )
                
                # Further reduce gamma to let agent explore blocking more freely
                self.gamma = max_r_eff_penalty / 100.0
                self.div_scale = min(np.median(fraud_limits) * 5, self.c * 1.5) * 2
            else:
                self.gamma = 100.0
                self.div_scale = 1.0
                max_v = 13.8 # Approx log1p of 1,000,000 NT$
                max_r_eff_penalty = 100.0
                
            self.env_bounds = {
                'eff': [-max_r_eff_penalty, max_v],
                'drift': [-self.gamma * 16.0, 0.0],
                'div': [0.0, self.div_scale * 2.0]
            }

        self.reset()

    def reset(self):
        self.current_step = 0
        self.window_labels = deque(maxlen=self.W)
        self.window_fps = deque(maxlen=self.W)
        self.window_negatives = deque(maxlen=self.W)
        
        self.mu_rejected = np.zeros(self.embeddings.shape[1])
        return self._get_state()

    def _get_state(self):
        if self.current_step >= self.n_samples:
            return np.zeros(self.emb_dim)
            
        base_emb = self.embeddings[self.current_step]        
        if self.mode == 'pareto':
            # Append continuous friction signal to state to resolve POMDP
            total_negatives = sum(self.window_negatives)
            current_fpr = sum(self.window_fps) / total_negatives if total_negatives > 0 else 0.0
            return np.append(base_emb, [current_fpr])
        return base_emb

    def step(self, action):
        if self.current_step >= self.n_samples:
            raise ValueError("Environment terminated.")
            
        y_t = self.labels[self.current_step]
        L_t = self.limits[self.current_step] # Credit Limit
        v_t = self.embeddings[self.current_step]
        
        # Metrics update
        self.window_labels.append(y_t)
        self.window_negatives.append(1 if y_t == 0 else 0)
        is_fp = 1 if (action == 1 and y_t == 0) else 0
        self.window_fps.append(is_fp)
        
        rho_W = max(sum(self.window_labels) / self.W, 0.001) if len(self.window_labels) == self.W else 0.1
        total_negatives = sum(self.window_negatives)
        fpr_W = sum(self.window_fps) / total_negatives if total_negatives > 0 else 0.0
        
        if self.mode == "scalar":
            p_t = float(action)
            # Log-Loss approximation for probability predictions
            reward = (y_t * math.log(p_t + self.epsilon)) + ((1 - y_t) * math.log(1 - p_t + self.epsilon))
        elif self.mode == "pareto":
            # 1. Financial Efficacy (r_eff) - ADAPTED FOR CREDIT
            if action == 1 and y_t == 1:     # TP: Rejected Defaulter
                r_eff = L_t * 25.0           # EXTREME BOOST: Force agent to find defaulters
            elif action == 0 and y_t == 1:   # FN: Approved Defaulter
                r_eff = -L_t * ((1 / rho_W)* 25.0) # EXTREME PENALTY: Never miss a defaulter
            elif action == 1 and y_t == 0:   # FP: Rejected Good Client
                r_eff = -self.c * 0.1        # MINIMAL PENALTY: Allow blocking aggressively
            else:                            # TN: Approved Good Client
                r_eff = L_t * 0.2 # Earned interest yield!
                                
            # 2. Friction Brake (r_drift)
            fpr_threshold = 0.10
            if is_fp == 1:
                # Cap the ratio to 4.0 to prevent gradient explosion
                ratio = min(fpr_W / fpr_threshold, 4.0) 
                r_drift = -self.gamma * (ratio ** 2)
            else:
                r_drift = 0.0
            
            # 3. Semantic Diversity (r_div)
            r_div = 0.0
            if action == 1 and y_t == 1: # If successfully rejected a defaulter
                norm_v = np.linalg.norm(v_t)
                norm_mu = np.linalg.norm(self.mu_rejected)
                if norm_mu > 0 and norm_v > 0:
                    cos_sim = np.dot(v_t, self.mu_rejected) / (norm_v * norm_mu)
                    r_div = (1.0 - cos_sim) * self.div_scale 
                else:
                    r_div = 1.0 * self.div_scale    
                
                self.mu_rejected = (self.alpha * v_t) + ((1 - self.alpha) * self.mu_rejected)
                
            reward = np.array([r_eff, r_drift, r_div], dtype=np.float32)
        else:
            raise ValueError("Invalid mode. Choose 'scalar' or 'pareto'.")

        self.current_step += 1
        return self._get_state(), reward, self.current_step >= self.n_samples, {}