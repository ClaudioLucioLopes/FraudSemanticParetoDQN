import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')

class CreditDataHandlerLongitudinal:
    def __init__(self, data_path, cache_dir="./cache_credit", model_name="all-MiniLM-L6-v2"):
        """
        Processes the UCI Credit dataset into longitudinal semantic embeddings.
        Variable mapping: X1-X5 (Demographics), X6-X11 (Status), X12-X17 (Bills), X18-X23 (Payments).
        """
        self.data_path = data_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Determine device with compatibility check
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                self.device = "cuda"
            else:
                print(f"Warning: GPU {torch.cuda.get_device_name()} (CC {major}.{minor}) "
                      f"is incompatible with this PyTorch build. Falling back to CPU.")
                self.device = "cpu"
        else:
            self.device = "cpu"

        print(f"Loading Semantic Encoder: {model_name} on {self.device}...")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        
        self.df = None
        self.embeddings = None

    @staticmethod
    def _get_longitudinal_state_narrative(row):
        """
        Constructs a target-free semantic string of the 6-month financial history.
        Essential for preventing target leakage in the state representation.
        """
        # Demographic Context (X1-X5)
        sex = "Male" if row['X2'] == 1 else "Female"
        edu = {1: "grad school", 2: "university", 3: "high school", 4: "others"}.get(row['X3'], "other")
        marriage = {1: "married", 2: "single", 3: "others"}.get(row['X4'], "other")
        
        # Longitudinal mapping (X6-X23)
        months = ["April", "May", "June", "July", "August", "September"]
        pay_status_vars = ["X11", "X10", "X9", "X8", "X7", "X6"] 
        bill_amt_vars   = ["X17", "X16", "X15", "X14", "X13", "X12"]
        prev_pay_vars   = ["X23", "X22", "X21", "X20", "X19", "X18"]

        history = []
        for i, month in enumerate(months):
            s = int(row[pay_status_vars[i]])
            status_desc = "on time" if s <= 0 else f"{s} month(s) late"
            history.append(
                f"{month}: status {status_desc}, bill NT${row[bill_amt_vars[i]]:.0f}, paid NT${row[prev_pay_vars[i]]:.0f}"
            )

        # Build state narrative without including the target 'Y'
        return (
            f"Client {int(row['ID'])}: {sex}, {edu}, {marriage}, age {int(row['X5'])}. "
            f"Credit Limit: NT${row['X1']:.0f}. "
            f"6-Month Behavioral Sequence: " + " | ".join(history) + "."
        )

    def load_and_preprocess(self, use_cache=True):
        """Loads dataset and performs longitudinal narrative generation."""
        cache_file = os.path.join(self.cache_dir, "processed_credit.pkl")
        if use_cache and os.path.exists(cache_file):
            print("Loading preprocessed data from cache...")
            self.df = pd.read_pickle(cache_file)
            return self.df

        # Load CSV (Ensure file has columns ID, X1...X23, Y)
        print(f"Loading dataset from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Ensure all columns exist; if names differ from X1, rename accordingly
        # The UCI dataset often uses labels like 'LIMIT_BAL' etc. 
        # Here we assume the user has standardized them to X1-X23 as requested.
        
        self.df.to_pickle(cache_file)
        return self.df

    def generate_embeddings(self, use_cache=True):
        """Generates dense vectors from longitudinal narratives."""
        if self.df is None: 
            raise ValueError("Run load_and_preprocess() before generating embeddings.")

        cache_file = os.path.join(self.cache_dir, "credit_embeddings.npy")
        if use_cache and os.path.exists(cache_file):
            print("Loading embeddings from cache...")
            self.embeddings = np.load(cache_file)
            return self.embeddings

        print("Encoding longitudinal history (this captures the behavioral trajectory)...")
        semantic_strings = self.df.apply(self._get_longitudinal_state_narrative, axis=1).tolist()
        
        self.embeddings = self.encoder.encode(
            semantic_strings,
            show_progress_bar=True,
            batch_size=128
        )

        np.save(cache_file, self.embeddings)
        return self.embeddings

    def get_train_test_split(self, train_ratio=0.8):
        """
        Splits data into train and test sets for RL and Supervised tasks.
        Maintains order to simulate real-world inference.
        """
        if self.embeddings is None:
            raise ValueError("Run generate_embeddings() before splitting.")

        split_idx = int(len(self.df) * train_ratio)

        # Labels (Y) and context values (X1 - Limit) for reward functions
        labels = self.df['Y'].values
        limits = self.df['X1'].values

        train_data = {
            'embeddings': self.embeddings[:split_idx],
            'labels':     labels[:split_idx],
            'limits':     limits[:split_idx],
        }
        test_data = {
            'embeddings': self.embeddings[split_idx:],
            'labels':     labels[split_idx:],
            'limits':     limits[split_idx:],
        }

        print(f"Split completed: {len(train_data['labels'])} train / {len(test_data['labels'])} test instances.")
        return train_data, test_data

# # ------------------------------------------------------------------
# # Testing the Implementation
# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     # 1. Create dummy data to simulate the UCI CSV structure
#     data = {
#         'ID': range(10),
#         'X1': [50000]*10, 'X2': [1,2]*5, 'X3': [1,2,3,4,1,2,3,4,1,2], 
#         'X4': [1,2,1,2,1,2,1,2,1,2], 'X5': [30]*10,
#         'X6': [0]*10, 'X7': [0]*10, 'X8': [0]*10, 'X9': [0]*10, 'X10': [0]*10, 'X11': [0]*10,
#         'X12': [100]*10, 'X13': [100]*10, 'X14': [100]*10, 'X15': [100]*10, 'X16': [100]*10, 'X17': [100]*10,
#         'X18': [50]*10, 'X19': [50]*10, 'X20': [50]*10, 'X21': [50]*10, 'X22': [50]*10, 'X23': [50]*10,
#         'Y': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
#     }
#     dummy_csv = "./data/test_credit.csv"
#     pd.DataFrame(data).to_csv(dummy_csv, index=False)

#     # 2. Instantiate and run handler
#     handler = CreditDataHandlerLongitudinal(data_path=dummy_csv)
#     handler.load_and_preprocess(use_cache=False)
#     handler.generate_embeddings(use_cache=False)
    
#     # 3. Perform split
#     train, test = handler.get_train_test_split(train_ratio=0.7)

#     # 4. Assertions for validation
#     print("\n--- Test Results ---")
#     print(f"Train Embedding Shape: {train['embeddings'].shape}") # Expected (7, 384)
#     print(f"Test Labels Count: {len(test['labels'])}")           # Expected 3
#     assert train['embeddings'].shape[0] == 7
#     assert 'labels' in train and 'limits' in train
#     print("Test passed successfully.")
    
#     # Cleanup
#     os.remove(dummy_csv)