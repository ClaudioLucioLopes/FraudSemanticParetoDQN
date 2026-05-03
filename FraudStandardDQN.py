import torch
import numpy as np
import random
import os
import pandas as pd 
from collections import deque
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, roc_auc_score
from FraudDataHandlerImproved import FraudDataHandlerImproved
from FraudMOEnv import FraudMOEnv
from StandardDQN import StandardDQNAgent
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

if __name__ == "__main__":
    # 1. Load Data & Initialize Environment
    handler = FraudDataHandlerImproved(data_dir="./data")
    handler.load_and_preprocess()
    handler.generate_embeddings()
    train_data, test_data = handler.get_train_test_split()
    
    # Initialize Environment in "scalar" Mode A
    env = FraudMOEnv(train_data, mode="scalar")
    state_dim = env.emb_dim
    agent = StandardDQNAgent(state_dim=state_dim)
    
    print("\n--- Training Standard DQN (Mode A: Scalar Log-Loss) ---")
    episodes = 20 # Sequential dataset pass
    batch_size = 256
    
    history = []  # Tracking episodic returns

    
    for ep in range(episodes):
        state = env.reset()
        print(f"Starting training {ep+1} of {episodes} episodes...")

        done = False
        step_count = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            
            if step_count % 4 == 0:
                agent.train_step(batch_size)
            step_count += 1
            
        agent.epsilon = max(0.05, agent.epsilon * 0.85)
        agent.target_net.load_state_dict(agent.q_net.state_dict())

        print(f"[ Ep {ep+1:>2}/{episodes}] "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"reward: {reward:>8.1f}")
        
   

    print("\n--- Evaluation on Test Set ---")
    test_env = FraudMOEnv(test_data, mode="scalar")
    metric_test_env = FraudMOEnv(test_data, mode="pareto")
    
    state = test_env.reset()
    metric_test_env.reset()
    done = False
    
    y_test, y_pred, y_probs, embeddings_list = [], [], [], []
    vec_rewards = []
    
    while not done:
        idx = test_env.current_step
        y_test.append(test_env.labels[idx])
        # Capture the embedding for this state
        embeddings_list.append(state)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.q_net(state_tensor).squeeze().cpu().numpy()
            
        action = np.argmax(q_values)
        y_pred.append(action)
        
        prob = np.exp(q_values) / np.sum(np.exp(q_values))
        y_probs.append(prob[1]) 
        
        state, _, done, _ = test_env.step(action)
        _, vec_reward, _, _ = metric_test_env.step(action)
        vec_rewards.append(vec_reward)

    # --- Metrics & Semantic Diversity ---
    embeddings_arr = np.array(embeddings_list)
    tp_indices = np.where((np.array(y_test) == 1) & (np.array(y_pred) == 1))[0]
    
    cov_trace = np.trace(np.cov(embeddings_arr[tp_indices], rowvar=False)) if len(tp_indices) > 1 else 0.0

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    
    df_eval_history = pd.DataFrame(vec_rewards, columns=['r_eff', 'r_drift', 'r_div'])
    
    costs = -df_eval_history[['r_eff', 'r_drift', 'r_div']].values
    nds = NonDominatedSorting()
    fronts = nds.do(costs)
    if len(fronts) > 0:
        non_dominated_indices = fronts[0]
        df_non_dominated_eval = df_eval_history.iloc[non_dominated_indices]
    else:
        df_non_dominated_eval = df_eval_history
        
    os.makedirs("./results", exist_ok=True)
    df_non_dominated_eval.to_csv("./results/FraudStandardDQN_eval_nondominated.csv", index=False)

    # --- Artifact Generation (Matches FraudBaselineXGB.py) ---
    os.makedirs("./results", exist_ok=True)
    pd.DataFrame({"y_true": y_test, "y_prob": y_probs, "y_pred": y_pred}).to_csv("./results/FraudStandardDQN_test_predictions.csv", index=False)
    results_df = pd.DataFrame([{
            "Model": "Standard DQN",
            "Precision": prec, "Recall": rec, "F1-Score": f1,
            "PR-AUC": pr_auc, "AU-ROC": roc_auc, "FPR": fpr,
            "Div Trace": cov_trace,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }])


    results_df.to_csv("./results/FraudBaselineXGB_benchmark_results.csv", mode='a', header=not os.path.exists("./results/FraudBaselineXGB_benchmark_results.csv"), index=False)
