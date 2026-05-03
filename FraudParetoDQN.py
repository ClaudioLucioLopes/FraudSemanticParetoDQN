if __name__ == "__main__":
    from FraudDataHandlerImproved import FraudDataHandlerImproved
    from FraudDataHandler import FraudDataHandler
    from FraudMOEnv import FraudMOEnv
    from ParetoDQN import ParetoFraudAgent
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, roc_auc_score
    import torch
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    
    print("\n[MORL System] Initializing Data Pipeline...")
    handler = FraudDataHandlerImproved(data_dir="./data")
    handler.load_and_preprocess()
    handler.generate_embeddings()
    train_data, test_data = handler.get_train_test_split()
    
    print("\n[MORL System] Instantiating Semantic MOMDP Environment...")
    env = FraudMOEnv(train_data, mode="pareto")
    
    device = "cpu"
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 7:
            device = "cuda"
        else:
            print(f"[MORL System] GPU {torch.cuda.get_device_name()} (CC {major}.{minor}) "
                  f"is incompatible. Falling back to CPU for stability.")
    
    print(f"[MORL System] Initializing Agent on {device}...")
    agent = ParetoFraudAgent(env_bounds=env.env_bounds, state_dim=env.emb_dim, num_objectives=3, device=device)
    
    print("\n[MORL System] Commencing Pareto-DQN Training Phase...")
    episodes = 20
    batch_size = 256
  
    for ep in range(episodes):
        state = env.reset()
        done = False
        step_count = 0
        ep_rewards = []
        print(f"Starting training {ep+1} of {episodes} episodes...")

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_rewards.append(reward)
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            
            if step_count % (batch_size) == 0:
                agent.train_step(batch_size)
                # agent.update_epsilon()
                agent.epsilon = max(0.05, agent.epsilon * 0.9985)


            if step_count > 0 and step_count % 10000 == 0:
                ep_rewards_np = np.array(ep_rewards)
                total_r_eff = np.sum(ep_rewards_np[:, 0])
                total_r_drift = np.sum(ep_rewards_np[:, 1])
                total_r_div = np.sum(ep_rewards_np[:, 2])
                print(f"    -> [Ep {ep+1} | Step {step_count}/{env.n_samples}] "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Current Return: r_eff: {total_r_eff:>8.1f} | "
                      f"r_drift: {total_r_drift:>8.1f} | "
                      f"r_div: {total_r_div:>6.2f}")
                
            step_count += 1
            
        agent.update_epsilon()
        
        # Pareto-DQN specific target network updates
        # Safely get state dicts depending on your architecture
        if hasattr(agent, 'rew_opt') and not isinstance(agent.rew_net, torch.nn.Module):
            pass 
        elif hasattr(agent.rew_net, 'state_dict'):
            agent.rew_target.load_state_dict(agent.rew_net.state_dict())
        agent.nd_target.load_state_dict(agent.nd_net.state_dict())
        

 
    print("\n[MORL System] Evaluating Pareto-DQN on Test Set...")
    train_calib = {
        'c': env.c,
        'gamma': env.gamma,
        'div_scale': env.div_scale,
        'env_bounds': env.env_bounds
    }
    test_env = FraudMOEnv(test_data, mode="pareto", calib=train_calib)
    state = test_env.reset()
    done = False
    
    y_true, y_pred, y_probs, tp_embeddings = [], [], [], []
    vec_rewards = []
    
    while not done:
        actual_label = test_env.labels[test_env.current_step]
        y_true.append(actual_label)
        
        # Compute action and softmax-hypervolume probability in one fast pass
        action, prob = agent.select_action(state, inference=True, n_samples=5, return_prob=True)
        y_pred.append(action)
        
        if actual_label == 1 and action == 1:
            tp_embeddings.append(state.copy())
            
        y_probs.append(prob)
        
        state, vec_reward, done, _ = test_env.step(action)
        vec_rewards.append(vec_reward)

    # --- Metrics & Semantic Diversity ---
   
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_true, y_probs)
    cov_trace = np.trace(np.cov(np.array(tp_embeddings), rowvar=False)) if len(tp_embeddings) > 1 else 0.0

   
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
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
    df_non_dominated_eval.to_csv("./results/FraudParetoDQN_eval_nondominated.csv", index=False)

    # --- Artifact Generation (Matches BaselineXGB) ---
    pd.DataFrame({"y_true": y_true, "y_prob": y_probs, "y_pred": y_pred}).to_csv(
        "./results/FraudParetoDQN_test_predictions.csv", index=False)
        
    results_df = pd.DataFrame([{
            "Model": "Pareto-DQN",
            "Precision": prec, "Recall": rec, "F1-Score": f1,
            "PR-AUC": pr_auc, "AU-ROC": roc_auc, "FPR": fpr,
            "Div Trace": cov_trace,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }])


    
    results_df.to_csv("./results/FraudBaselineXGB_benchmark_results.csv", 
                      mode='a', header=False, index=False)
    
    print(f"\n[+] Benchmark Results Appended.\n{results_df.to_markdown(index=False)}")
