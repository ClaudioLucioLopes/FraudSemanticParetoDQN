import xgboost as xgb
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix, roc_auc_score
from CreditDataHandler import CreditDataHandlerLongitudinal

class BaselineXGB:
    def __init__(self, random_state=42):
        """
        Initializes the static supervised baseline.
        The objective is binary:logistic, outputting probabilities that minimize log-loss.
        """
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=random_state,
            eval_metric='logloss'
        )

    def train(self, X_train, y_train):
        """
        Trains the XGBoost model on the provided feature space.
        """
        print(f"Training XGBoost on {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test, X_test_emb, feature_type="Semantic"):
        """
        Evaluates the model and generates artifacts for the scientific report.
        X_test_emb is passed explicitly to calculate the Semantic Diversity of Caught Defaults.
        """
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # --- Generate Structured Metrics ---
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # --- Semantic Diversity Metric ---
        # Extract the semantic embeddings corresponding to the True Positives (caught defaults)
        tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
        
        if len(tp_indices) > 1:
            tp_embeddings = X_test_emb[tp_indices]
            cov_matrix = np.cov(tp_embeddings, rowvar=False)
            cov_trace = np.trace(cov_matrix)
        else:
            cov_trace = 0.0

        #Export Raw Predictions for Plotting (ROC/PR Curves)
        file_name_safe = feature_type.replace(" ", "_")
        pd.DataFrame({
            "y_true": y_test,
            "y_prob": y_prob,
            "y_pred": y_pred
        }).to_csv(f"./results/CreditBaselineXGB_{file_name_safe}_test_predictions.csv", index=False)

        #Append to Final Table
        results_df = pd.DataFrame([{
            "Model": f"XGBoost ({feature_type})",
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
        
        file_path = "./results/CreditBaselineXGB_benchmark_results.csv"
        results_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        
        print(f"\n[+] Table Row Appended Successfully ({feature_type}):")
        print(results_df.to_markdown(index=False))
        
        return pr_auc

def extract_raw_features(df):
    """
    Extracts the raw tabular features (X1 to X23) for the supervised baseline.
    Removes ID and the target variable Y to prevent data leakage.
    """
    cols_to_drop = ['ID', 'Y']
    X_raw = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    X_raw = X_raw.select_dtypes(include=[np.number])
    return X_raw.values

if __name__ == "__main__":
    print("\n[MORL Baseline] Initializing Data Pipeline...")
    handler = CreditDataHandlerLongitudinal(data_path="./results/credit_card_clients.csv")
    df = handler.load_and_preprocess()
    embeddings = handler.generate_embeddings()
    train_data, test_data = handler.get_train_test_split()
    
    split_idx = len(train_data['labels'])
    
    # ---------------------------------------------------------
    # Baseline 1: Without Embeddings (Raw Tabular History X1-X23)
    # ---------------------------------------------------------
    print("\n[MORL Baseline] Executing Baseline 1 (Raw Tabular)...")
    X_raw = extract_raw_features(df)
    X_train_raw = X_raw[:split_idx]
    X_test_raw = X_raw[split_idx:]
    
    xgb_raw = BaselineXGB()
    xgb_raw.train(X_train_raw, train_data['labels'])
    # Pass test_data['embeddings'] explicitly to compute Diversity Trace of caught defaults
    xgb_raw.evaluate(X_test_raw, test_data['labels'], test_data['embeddings'], feature_type="Raw Tabular")
    
    # ---------------------------------------------------------
    # Baseline 2: With Semantic Embeddings (Vectorized Narratives)
    # ---------------------------------------------------------
    print("\n[MORL Baseline] Executing Baseline 2 (Semantic Embeddings)...")
    X_train_emb = train_data['embeddings']
    X_test_emb = test_data['embeddings']
    
    xgb_emb = BaselineXGB()
    xgb_emb.train(X_train_emb, train_data['labels'])
    xgb_emb.evaluate(X_test_emb, test_data['labels'], test_data['embeddings'], feature_type="Semantic Embeddings")
    
    # ---------------------------------------------------------
    # Baseline 3: Combined (Raw Tabular + Semantic Embeddings)
    # ---------------------------------------------------------
    print("\n[MORL Baseline] Executing Baseline 3 (Combined Features)...")
    X_train_combined = np.hstack((X_train_raw, X_train_emb))
    X_test_combined = np.hstack((X_test_raw, X_test_emb))
    
    xgb_combined = BaselineXGB()
    xgb_combined.train(X_train_combined, train_data['labels'])
    xgb_combined.evaluate(X_test_combined, test_data['labels'], test_data['embeddings'], feature_type="Combined Features")
    
    print("\n[+] XGBoost benchmarks complete. Check benchmark_results.csv and test prediction CSVs.")