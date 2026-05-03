import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix,roc_auc_score
from FraudDataHandler import FraudDataHandler
from FraudDataHandlerImproved import FraudDataHandlerImproved
import os


class BaselineXGB:
    def __init__(self, random_state=42):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )

    def train(self, X_train, y_train):
        print(f"Training XGBoost on {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test, X_test_emb, feature_type="Semantic"):
        """
        Updated evaluation to include Semantic Diversity (Div Trace) 
        and structured artifact generation.
        """
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # --- Metrics ---
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # --- Semantic Diversity Metric (Trace of Covariance) ---
        tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
        if len(tp_indices) > 1:
            cov_matrix = np.cov(X_test_emb[tp_indices], rowvar=False)
            cov_trace = np.trace(cov_matrix)
        else:
            cov_trace = 0.0

        # --- Artifact Generation ---
        os.makedirs("./results", exist_ok=True)
        file_name_safe = feature_type.replace(" ", "_")
        pd.DataFrame({
            "y_true": y_test, "y_prob": y_prob, "y_pred": y_pred
        }).to_csv(f"./results/FraudBaselineXGB_{file_name_safe}_test_predictions.csv", index=False)

        results_df = pd.DataFrame([{
            "Model": f"XGBoost ({feature_type})",
            "Precision": prec, "Recall": rec, "F1-Score": f1,
            "PR-AUC": pr_auc, "AU-ROC": roc_auc, "FPR": fpr,
            "Div Trace": cov_trace,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        }])
        
        results_df.to_csv("./results/FraudBaselineXGB_benchmark_results.csv", 
                          mode='a', header=not os.path.exists("./results/FraudBaselineXGB_benchmark_results.csv"), 
                          index=False)
        
        print(f"\n[+] Table Row Appended ({feature_type}):")
        print(results_df.to_markdown(index=False))
        
        return pr_auc

def extract_raw_features(df):
    """
    Helper function to extract numerical/encoded raw features for the 'without embeddings' baseline.
    """
    # Select numerical columns and encode categoricals for basic XGBoost compatibility
    features = ['purchase_value', 'age', 'velocity_seconds']
    X_raw = df[features].copy()
    
    # Simple frequency encoding for categorical variables as a basic representation
    for col in ['source', 'browser', 'sex', 'country']:
        freq_encoding = df[col].value_counts(normalize=True)
        X_raw[col + '_freq'] = df[col].map(freq_encoding)
        
    return X_raw.values

if __name__ == "__main__":
    # 1. Initialize Data Handler and load data
    # handler = FraudDataHandler(data_dir="./data")
    handler = FraudDataHandlerImproved(data_dir="./data")
    df = handler.load_and_preprocess()
    
    # 2. Generate Semantic Embeddings (v_t)
    embeddings = handler.generate_embeddings()
    
    # 3. Retrieve split dictionaries
    train_data, test_data = handler.get_train_test_split()
    
    # Split index used in DataHandler to ensure temporal consistency
    split_idx = len(train_data['labels'])
    
   # --- Baseline 1: Without Embeddings (Raw Tabular Features) ---
    X_raw = extract_raw_features(df)
    X_train_raw = X_raw[:split_idx]
    X_test_raw = X_raw[split_idx:]
    
    xgb_raw = BaselineXGB()
    xgb_raw.train(X_train_raw, train_data['labels'])
    # Added test_data['embeddings'] as the 3rd argument
    xgb_raw.evaluate(X_test_raw, test_data['labels'], test_data['embeddings'], feature_type="Raw Tabular")
    
    # --- Baseline 2: With Semantic Embeddings (v_t) ---
    X_train_emb = train_data['embeddings']
    X_test_emb = test_data['embeddings']
    
    xgb_emb = BaselineXGB()
    xgb_emb.train(X_train_emb, train_data['labels'])
    # Added test_data['embeddings'] as the 3rd argument
    xgb_emb.evaluate(X_test_emb, test_data['labels'], test_data['embeddings'], feature_type="Semantic Embeddings")
    
