import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, roc_auc_score
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Ensure the import matches your actual file structure
from CreditDataHandler import CreditDataHandlerLongitudinal
from CreditMOEnv import CreditMOEnv
from StandardDQN import StandardDQNAgent

if __name__ == "__main__":
    print("\n--- Initializing Credit Risk Pipeline ---")
    handler = CreditDataHandlerLongitudinal(data_path="./data/credit_card_clients.csv")
    handler.load_and_preprocess()
    handler.generate_embeddings()
    train_data, test_data = handler.get_train_test_split()
    
    env = CreditMOEnv(train_data, mode="scalar")
    state_dim = env.emb_dim
    agent = StandardDQNAgent(state_dim=state_dim)
    
    print("\n--- Training Standard DQN (Mode A: Scalar Log-Loss) ---")
    episodes = 20
    batch_size = 256
    
    # Track episodic metrics for Learning Curves
    training_logs = []

    history = []  # Tracking episodic returns

    metric_env = CreditMOEnv(train_data, mode="pareto")
    
    for ep in range(episodes):
        state = env.reset()
        print(f"Starting training {ep+1} of {episodes} episodes...")

        done = False
        step_count = 0
        ep_rewards = []
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            ep_rewards.append(reward)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            
            if step_count % 4 == 0:
                agent.train_step(batch_size)
                
            step_count += 1
            
        # Aggressive manual epsilon decay
        agent.epsilon = max(0.05, agent.epsilon * 0.85)
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        
        total_reward = np.sum(ep_rewards)
        avg_reward = np.mean(ep_rewards)
        print(f"Episode {ep+1} Complete. Epsilon: {agent.epsilon:.3f} | Total Reward: {total_reward:.2f}")
        
        training_logs.append({
            "Episode": ep + 1,
            "Epsilon": agent.epsilon,
            "Total_Reward": total_reward,
            "Avg_Reward": avg_reward
        })


        
    # Export Training Logs
    pd.DataFrame(training_logs).to_csv("StandardDQN_training_logs.csv", index=False)
    print("\n[+] Saved training logs to StandardDQN_training_logs.csv")
        

    print("\n--- Evaluation on Test Set ---")
    calib_params = {'c': env.c, 'gamma': env.gamma, 'div_scale': env.div_scale}
    test_env = CreditMOEnv(test_data, mode="scalar", calib=calib_params)
    metric_test_env = CreditMOEnv(test_data, mode="pareto", calib=calib_params)
    
    state = test_env.reset()
    metric_test_env.reset()
    done = False
    
    y_true, y_pred, y_probs = [], [], []
    tp_embeddings = [] 
    
    vec_rewards = []
    state_trajectory = [state]
    
    while not done:
        actual_label = test_env.labels[test_env.current_step]
        y_true.append(actual_label)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.q_net(state_tensor).squeeze().cpu().numpy()
            
        action = np.argmax(q_values)
        y_pred.append(action)
        
        if actual_label == 1 and action == 1:
            tp_embeddings.append(state.copy())
        
        prob = np.exp(q_values) / np.sum(np.exp(q_values))
        y_probs.append(prob[1]) 
        
        state, _, done, _ = test_env.step(action)
        _, vec_reward, _, _ = metric_test_env.step(action)
        
        vec_rewards.append(vec_reward)
        if not done:
            state_trajectory.append(state)
            
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
    df_non_dominated_eval.to_csv("./results/CreditStandardDQN_eval_nondominated.csv", index=False)
       
    #  Export Raw Predictions for Plotting (ROC/PR Curves)
    pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_probs,
        "y_pred": y_pred
    }).to_csv("./results/CreditStandardDQN_test_predictions.csv", index=False)
    print("[+] Saved raw test predictions to StandardDQN_test_predictions.csv")

    # --- Generate Structured Metrics ---
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_true, y_probs) # Calculated AU-ROC
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    if len(tp_embeddings) > 1:
        cov_matrix = np.cov(np.array(tp_embeddings), rowvar=False)
        cov_trace = np.trace(cov_matrix)
    else:
        cov_trace = 0.0

    # [ARTIFACT 3] Append to Final Table with exactly the requested columns
    results_df = pd.DataFrame([{
        "Model": "Standard DQN",
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "PR-AUC": pr_auc,
        "AU-ROC": roc_auc,
        "FPR": fpr,
        "Div Trace": cov_trace,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }])
    
    file_path = "./results/CreditStandardDQN_benchmark_results.csv"
    # Ensure header is written if file doesn't exist, otherwise append without header
    results_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    
    print("\n[+] Table Row Appended Successfully:")
    print(results_df.to_markdown(index=False))