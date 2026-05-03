import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')

class FraudDataHandler:
    def __init__(self, data_dir="./data", cache_dir="./cache", model_name="all-MiniLM-L6-v2"):
        """
        Handles data ingestion, feature engineering, and semantic embedding.
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.fraud_path = f"{data_dir}/Fraud_Data.csv"
        self.ip_path = f"{data_dir}/IpAddress_to_Country.csv"
        
        # Create directories if they don't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine device with compatibility check
        if torch.cuda.is_available():
            # Current PyTorch build supports CC 7.5+ (Turing/Ampere/etc.)
            # Pascal (10-series like 1050 Ti) is CC 6.1
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                self.device = "cuda"
            else:
                print(f"Warning: GPU {torch.cuda.get_device_name()} (CC {major}.{minor}) "
                      f"is incompatible with this PyTorch build. Falling back to CPU.")
                self.device = "cpu"
        else:
            self.device = "cpu"

        # Load the semantic encoder
        print(f"Loading Semantic Encoder: {model_name} on {self.device}...")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        
        self.df = None
        self.embeddings = None

    def _map_ip_to_country(self, df_fraud, df_countries):
        """
        Maps numerical IP addresses to countries using vectorized interval matching.
        """
        print("Mapping IP addresses to countries (optimized)...")
        
        # Sort both for merge_asof
        df_fraud = df_fraud.sort_values('ip_address')
        df_countries = df_countries.sort_values('lower_bound_ip_address')
        
        # merge_asof finds the nearest lower_bound_ip_address that is <= ip_address
        df_mapped = pd.merge_asof(
            df_fraud, 
            df_countries, 
            left_on='ip_address', 
            right_on='lower_bound_ip_address',
            direction='backward'
        )
        
        # Validate that the IP is actually within the range
        mask = (df_mapped['ip_address'] >= df_mapped['lower_bound_ip_address']) & \
               (df_mapped['ip_address'] <= df_mapped['upper_bound_ip_address'])
        
        df_mapped.loc[~mask, 'country'] = "Undefined"
        
        # Cleanup extra columns from countries df
        df_mapped = df_mapped.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
        
        return df_mapped

    def load_and_preprocess(self, use_cache=True):
        """
        Loads CSVs, engineers temporal features, and sorts chronologically.
        """
        cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
        if use_cache and os.path.exists(cache_file):
            print(f"Loading preprocessed data from cache: {cache_file}")
            self.df = pd.read_pickle(cache_file)
            return self.df

        print("Loading datasets...")
        df_fraud = pd.read_csv(self.fraud_path)
        df_countries = pd.read_csv(self.ip_path)

        # 1. Temporal Feature Engineering: Signup-to-Purchase Delta
        df_fraud['signup_time'] = pd.to_datetime(df_fraud['signup_time'])
        df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'])
        
        # Calculate velocity in seconds
        df_fraud['velocity_seconds'] = (df_fraud['purchase_time'] - df_fraud['signup_time']).dt.total_seconds()
        
        # 2. Chronological Sorting for RL Sequential Integrity
        # It is critical that the environment steps through transactions historically
        df_fraud = df_fraud.sort_values(by='purchase_time').reset_index(drop=True)
        
        # 3. Map IP to Country
        self.df = self._map_ip_to_country(df_fraud, df_countries)
        
        # 4. Enrich with more granular temporal features
        self.df['purchase_day_of_week'] = self.df['purchase_time'].dt.day_name()
        self.df['purchase_day_of_month'] = self.df['purchase_time'].dt.day.astype(str)
        self.df['purchase_month_name'] = self.df['purchase_time'].dt.month_name()

        # Sort back to chronological order (IP mapping might have reordered)
        self.df = self.df.sort_values(by='purchase_time').reset_index(drop=True)
        
        # Save to cache
        cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
        self.df.to_pickle(cache_file)
        
        print("Preprocessing complete and cached.")
        return self.df

    def generate_embeddings(self, use_cache=True):
        """
        Concatenates features into a semantic string and generates continuous embeddings.
        """
        if self.df is None:
            raise ValueError("Run load_and_preprocess() before generating embeddings.")

        cache_file = os.path.join(self.cache_dir, "embeddings.npy")
        if use_cache and os.path.exists(cache_file):
            print(f"Loading embeddings from cache: {cache_file}")
            self.embeddings = np.load(cache_file)
            return self.embeddings

        print("Generating semantic embeddings (this may take a while)...")
        
        # Features to include in the semantic string
        feature_columns = [
            'user_id', 'purchase_value', 'device_id', 'source', 
            'browser', 'sex', 'age', 'ip_address', 'country', 'velocity_seconds',
            'purchase_day_of_week', 'purchase_day_of_month', 'purchase_month_name'
        ]
        

        # Create a combined string representation for each transaction
        combined_text = self.df[feature_columns].astype(str).agg(' '.join, axis=1).tolist()
        
        # Generate d=384 dimensional vectors
        self.embeddings = self.encoder.encode(combined_text, show_progress_bar=True)
        
        # Save to cache
        cache_file = os.path.join(self.cache_dir, "embeddings.npy")
        np.save(cache_file, self.embeddings)
        
        return self.embeddings

    def get_train_test_split(self, train_ratio=0.8):
        """
        Splits the data chronologically to maintain causality for the MDP.
        Returns dictionaries for train and test sets containing features, embeddings, labels, and values.
        """
        if self.embeddings is None:
            raise ValueError("Run generate_embeddings() before splitting.")

        split_idx = int(len(self.df) * train_ratio)

        # Labels (0: Legitimate, 1: Fraud) and Monetary Values
        labels = self.df['class'].values
        values = self.df['purchase_value'].values

        # Split Embeddings
        X_train_emb = self.embeddings[:split_idx]
        X_test_emb = self.embeddings[split_idx:]

        # Split Labels
        y_train = labels[:split_idx]
        y_test = labels[split_idx:]
        
        # Split Values (required for r_eff computation)
        val_train = values[:split_idx]
        val_test = values[split_idx:]

        print(f"Data split chronologically: {len(y_train)} train, {len(y_test)} test instances.")

        train_data = {'embeddings': X_train_emb, 'labels': y_train, 'values': val_train}
        test_data = {'embeddings': X_test_emb, 'labels': y_test, 'values': val_test}

        return train_data, test_data

if __name__ == "__main__":
    # Quick test of the pipeline
    handler = FraudDataHandler(data_dir="./data")
    handler.load_and_preprocess()
    handler.generate_embeddings()
    train, test = handler.get_train_test_split()