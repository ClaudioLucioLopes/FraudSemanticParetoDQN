import numpy as np
import math
from collections import deque

class FraudMOEnv:
    def __init__(self, data, mode="pareto", W=1000, alpha=0.1, calib=None):
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        # self.values = data['values']
        self.raw_values = data['values']
        self.values = np.log1p(self.raw_values)
        
        self.mode = mode
        self.n_samples = len(self.labels)
        if mode == "pareto":
            self.emb_dim = self.embeddings.shape[1] +1 #  +1 for FPR state
        else:
            self.emb_dim = self.embeddings.shape[1]
        
        self.W = W
        self.alpha = alpha
        self.epsilon = 1e-5 

        # ---------------------------------------------------------
        # ENVIRONMENT BOUNDARY CALIBRATION
        # ---------------------------------------------------------
        if calib is not None:
            # INHERIT CALIBRATION (For Test Set)
            self.c = calib['c']
            self.gamma = calib['gamma']
            self.div_scale = calib['div_scale']
            self.env_bounds = calib['env_bounds']
            print(f"[Environment Calibration] Inherited Train Bounds -> c: {self.c:.2f}, gamma: {self.gamma:.2f}")
        else:
            # DYNAMIC CALIBRATION (For Train Set)
            # 1. Fetch legitimate transaction values directly from the database/dataset
            legit_values = self.values[self.labels == 0]
            fraud_values = self.values[self.labels == 1]
            
            # 2. Establish the baseline cost of insulting a customer
            self.c = (np.percentile(legit_values, 50) if len(legit_values) > 0 else 10.0)
            
            # Calibrate Gamma against the FN risk, not just 'c'
            avg_rho = max(len(fraud_values) / len(self.labels), 0.01) # e.g., global prevalence
            max_fraud_val = np.percentile(fraud_values, 99) if len(fraud_values) > 0 else 100.0

            #Multiply the FN penalty by 25.0 to make missing fraud mathematically terrifying
            max_fn_penalty = max_fraud_val * ( (1.0 / avg_rho) * 25.0 )
            
            #Weaken the friction brake divisor massively to allow more exploration of action=1
            self.gamma = max_fn_penalty / 200.0

            #Massively increase the reward for discovering new fraud typologies
            self.div_scale = self.c * 5.0

            # 2. FIX: Adjust the drift environment bound to match the cap
            self.env_bounds = {
                'eff': [-max_fn_penalty, max_fraud_val], 
                'drift': [-self.gamma * 16.0, 0.0], # 4 is max because ratio cap 
                'div': [0.0, self.div_scale * 2.0]
            }

            print(f"[Environment Calibration] Calculated -> c: {self.c:.2f}, gamma: {self.gamma:.2f}, div_scale: {self.div_scale:.2f}")
        self.reset()

    def reset(self):
        self.current_step = 0
        
        # Rolling metrics memory
        self.window_labels = deque(maxlen=self.W)
        self.window_fps = deque(maxlen=self.W)
        self.window_negatives = deque(maxlen=self.W)
        
        
        base_dim = self.embeddings.shape[1]
        self.mu_blocked = np.zeros(base_dim)
        
        return self._get_state()

    def _get_state(self):
        if self.mode=='pareto':
            if self.current_step >= self.n_samples:
                return np.zeros(self.emb_dim)
            base_emb = self.embeddings[self.current_step]        
            #Resolve POMDP by making the friction state observable
            total_negatives = sum(self.window_negatives)
            current_fpr = sum(self.window_fps) / total_negatives if total_negatives > 0 else 0.0
            # Augment the semantic vector with the continuous FPR signal
            return np.append(base_emb, [current_fpr])
        else:    
            if self.current_step >= self.n_samples:
                return np.zeros(self.emb_dim)
            return self.embeddings[self.current_step]

    def step(self, action):
        """
        Executes an action (0: Pass, 1: Block) and computes the reward.
        """
        if self.current_step >= self.n_samples:
            raise ValueError("Environment step called after episode termination.")
            
        y_t = self.labels[self.current_step]
        V_t = self.values[self.current_step]
        v_t = self.embeddings[self.current_step]
        
        # Update rolling metrics
        self.window_labels.append(y_t)
        self.window_negatives.append(1 if y_t == 0 else 0)
        is_fp = 1 if (action == 1 and y_t == 0) else 0
        self.window_fps.append(is_fp)
        
        # Calculate Rolling Ratios
        rho_W = max(sum(self.window_labels) / self.W, 0.001) if len(self.window_labels) == self.W else 0.1
        total_negatives = sum(self.window_negatives)
        fpr_W = sum(self.window_fps) / total_negatives if total_negatives > 0 else 0.0
        
        # ---------------------------------------------------------
        # MODE A: Standard DQN (Scalar Log-Loss)
        # ---------------------------------------------------------
        if self.mode == "scalar":
            p_t = float(action)
            # Negative Log-Loss: r_t = y*log(p) + (1-y)*log(1-p)
            reward = (y_t * math.log(p_t + self.epsilon)) + ((1 - y_t) * math.log(1 - p_t + self.epsilon))
            
        # ---------------------------------------------------------
        # MODE B: Pareto-DQN (Vectorial Multi-Objective)
        # ---------------------------------------------------------
        elif self.mode == "pareto":
            # 1. Financial Efficacy (r_eff)
            if action == 1 and y_t == 1:     # TP
                r_eff = V_t*40               # MASSIVE BOOST to force agent to block fraud
            elif action == 0 and y_t == 1:   # FN 
                r_eff = -V_t * ((1 / rho_W)*40) # MASSIVE PENALTY for missing fraud
            elif action == 1 and y_t == 0:   # FP
                r_eff = -self.c * 0.2        # WEAKEN the penalty for false positives
            else:                            # TN
                r_eff = V_t * 0.02
                
            # 2. Friction Brake (r_drift)
            fpr_threshold = 0.10
            if is_fp == 1:
                ratio = min(fpr_W / fpr_threshold, 2.0) 
                r_drift = -self.gamma * (ratio ** 2)
            else:
                r_drift = 0.0
           
            
            # 3. Semantic Diversity (r_div)
            r_div = 0.0
            if action == 1 and y_t == 1: # TP
                norm_v = np.linalg.norm(v_t)
                norm_mu = np.linalg.norm(self.mu_blocked)
                if norm_mu > 0 and norm_v > 0:
                    cos_sim = np.dot(v_t, self.mu_blocked) / (norm_v * norm_mu)
                    r_div = (1.0 - cos_sim) * self.div_scale 
                else:
                    r_div = 1.0 * self.div_scale    
                
                # Update EMA Centroid
                self.mu_blocked = (self.alpha * v_t) + ((1 - self.alpha) * self.mu_blocked)
                
            reward = np.array([r_eff, r_drift, r_div], dtype=np.float32)
        else:
            raise ValueError("Invalid mode. Choose 'scalar' or 'pareto'.")

        self.current_step += 1
        done = self.current_step >= self.n_samples
        next_state = self._get_state()
        
        return next_state, reward, done, {}

# =====================================================================
# Testing Suite
# =====================================================================
# if __name__ == "__main__":
#     print("Initializing Environment Tests...")
    
#     # Create dummy data representing 5 transactions with embedding dim=3
#     dummy_data = {
#         'embeddings': np.array([[1,0,0], [1,0,0], [0,1,0], [0.1, 0.9, 0], [0,0,1]]),
#         'labels': np.array([1, 1, 0, 1, 0]), # 1 is Fraud
#         'values': np.array([100, 50, 20, 200, 10])
#     }
    
#     # -------------------------------------------------
#     # Test 1: Scalar Mode (Log-Loss alignment)
#     # -------------------------------------------------
#     env_scalar = FraudMOEnv(dummy_data, mode="scalar", W=3)
#     state = env_scalar.reset()
    
#     print("\n--- TEST 1: Scalar Mode ---")
#     # Action 1 (Block) on True Fraud -> Should be ~0 reward (minimal loss)
#     _, r1, _, _ = env_scalar.step(action=1) 
#     print(f"Step 1 (TP) Scalar Reward: {r1:.4f}")
#     assert r1 > -1.0, "TP scalar reward should be close to 0."

#     # Action 0 (Pass) on True Fraud -> Should be large negative reward
#     _, r2, _, _ = env_scalar.step(action=0) 
#     print(f"Step 2 (FN) Scalar Reward: {r2:.4f}")
#     assert r2 < -10.0, "FN scalar reward should be heavily penalized."

#     # -------------------------------------------------
#     # Test 2: Pareto Mode (Vectorial Optimization)
#     # -------------------------------------------------
#     env_pareto = FraudMOEnv(dummy_data, mode="pareto", W=3, c=15.0)
#     state = env_pareto.reset()
    
#     print("\n--- TEST 2: Pareto Mode ---")
#     # Step 1: True Positive (Caught standard fraud)
#     # Expected: r_eff = +100, r_drift = 0, r_div = 1.0 (first item)
#     _, r1, _, _ = env_pareto.step(action=1)
#     print(f"Step 1 (TP) Vector Reward: {r1}")
#     assert r1[0] == 100 and r1[2] == 1.0, "Initial TP did not yield expected r_eff or r_div."
    
#     # Step 2: True Positive (Caught IDENTICAL fraud)
#     # Expected: r_div should be close to 0 because it matches the mu_blocked centroid
#     _, r2, _, _ = env_pareto.step(action=1)
#     print(f"Step 2 (TP - Similar) Vector Reward: {r2}")
#     assert r2[2] < 0.1, "Cosine distance for identical vector should be ~0."
    
#     # Step 3: False Positive (Insulted customer)
#     # Expected: r_eff = -15 (c), r_drift = Negative penalty based on FPR_W
#     _, r3, _, _ = env_pareto.step(action=1)
#     print(f"Step 3 (FP) Vector Reward: {r3}")
#     assert r3[0] == -15.0 and r3[1] < 0, "FP did not trigger friction penalties."
    
#     # Step 4: True Positive (Caught ORTHOGONAL fraud [0.1, 0.9, 0])
#     # Expected: r_div should be high because it's a new attack vector
#     _, r4, _, _ = env_pareto.step(action=1)
#     print(f"Step 4 (TP - Novel) Vector Reward: {r4}")
#     assert r4[2] > 0.5, "Orthogonal fraud vector did not yield a high discovery reward."

#     print("\nAll Environment tests passed successfully!")
