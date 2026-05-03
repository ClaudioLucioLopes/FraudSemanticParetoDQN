import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')


class FraudDataHandlerImproved:
    def __init__(self, data_dir="./data", cache_dir="./cache_improved", model_name="all-MiniLM-L6-v2"):
        """
        Handles data ingestion, feature engineering, and semantic embedding.
        Improved version with richer behavioral and temporal feature engineering.
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.fraud_path = f"{data_dir}/Fraud_Data.csv"
        self.ip_path = f"{data_dir}/IpAddress_to_Country.csv"

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_ip_to_country(self, df_fraud, df_countries):
        """Maps numerical IP addresses to countries using vectorized interval matching."""
        print("Mapping IP addresses to countries (optimized)...")

        df_fraud = df_fraud.sort_values('ip_address')
        df_countries = df_countries.sort_values('lower_bound_ip_address')

        df_mapped = pd.merge_asof(
            df_fraud,
            df_countries,
            left_on='ip_address',
            right_on='lower_bound_ip_address',
            direction='backward'
        )

        mask = (
            (df_mapped['ip_address'] >= df_mapped['lower_bound_ip_address']) &
            (df_mapped['ip_address'] <= df_mapped['upper_bound_ip_address'])
        )
        df_mapped.loc[~mask, 'country'] = "Undefined"
        df_mapped = df_mapped.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])

        return df_mapped

    def _add_temporal_features(self, df):
        """Extracts fine-grained temporal signals from purchase_time."""
        df['purchase_hour']       = df['purchase_time'].dt.hour
        df['purchase_day_of_week']= df['purchase_time'].dt.dayofweek          # 0=Mon…6=Sun
        df['purchase_day_name']   = df['purchase_time'].dt.day_name()
        df['purchase_day_of_month']= df['purchase_time'].dt.day
        df['purchase_month_name'] = df['purchase_time'].dt.month_name()
        df['is_weekend']          = df['purchase_day_of_week'].isin([5, 6]).astype(int)
        df['is_night_purchase']   = df['purchase_hour'].between(0, 6).astype(int)
        return df

    def _add_velocity_features(self, df):
        """
        Signup-to-purchase gap (velocity_seconds) and a human-readable bucket.
        Instant purchases (< 1 min) are the strongest single fraud indicator.
        """
        df['velocity_seconds'] = (
            df['purchase_time'] - df['signup_time']
        ).dt.total_seconds()

        df['velocity_bucket'] = pd.cut(
            df['velocity_seconds'],
            bins=[-1, 60, 3_600, 86_400, 604_800, float('inf')],
            labels=['instant', 'within_hour', 'within_day', 'within_week', 'slow']
        ).astype(str)

        return df

    def _add_device_features(self, df):
        """
        Computes device-level aggregations.
        Key insight: one device → many users is a strong fraud signal.
        """
        # Number of distinct users per device (computed over the full dataset)
        device_user_count = df.groupby('device_id')['user_id'].transform('nunique')
        df['device_user_count'] = device_user_count

        # Total purchases per device
        df['device_purchase_count'] = df.groupby('device_id')['device_id'].transform('count')

        return df

    def _add_user_behavioral_features(self, df):
        """
        Per-user sequential behavioral features respecting temporal causality.
        All rolling computations use shifted windows to avoid look-ahead leakage.
        """
        # Sort within each user by time
        df = df.sort_values(['user_id', 'purchase_time'])

        # Cumulative purchase index per user (0-based)
        df['user_purchase_count'] = df.groupby('user_id').cumcount()

        # Time since previous purchase by the same user (seconds)
        # Using groupby.diff() is vectorized and much faster
        df['time_since_last_purchase'] = (
            df.groupby('user_id')['purchase_time']
            .diff()
            .dt.total_seconds()
        )
        # NaN → first purchase for that user; fill with a large sentinel
        df['time_since_last_purchase'] = df['time_since_last_purchase'].fillna(-1)

        # Rolling spend over the last 3 purchases (shift(1) → no leakage)
        # Optimized: shift first, then rolling sum on the grouped object
        shifted_val = df.groupby('user_id')['purchase_value'].shift(1)
        df['user_rolling_spend_3'] = (
            shifted_val.groupby(df['user_id'])
            .rolling(window=3, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

        return df

    def _add_country_risk_feature(self, df):
        """
        Flags transactions from countries that appear rarely in the dataset.
        Rare = not in the top-10 most frequent countries.
        """
        top_countries = df['country'].value_counts().nlargest(10).index
        df['is_rare_country'] = (~df['country'].isin(top_countries)).astype(int)
        return df

    @staticmethod
    def _build_semantic_string(row):
        """
        Constructs a structured natural-language sentence for each transaction.
        SBERT was trained on sentences, not space-joined token dumps; this yields
        richer contextual embeddings than the naive join approach.

        Example: "User 22058 made purchase #1 worth $34.00 via SEO on Chrome. 
        Age 39, sex M, country Australia. Account signed up instant before this purchase. 
        Purchase at hour 2 (weekday, night). Device used by 1 distinct user(s). 
        Rolling spend over last 3 purchases: $0.00. Rare country: no."
        """
        return (
            f"User {row['user_id']} made purchase #{int(row['user_purchase_count']) + 1} "
            f"worth ${row['purchase_value']:.2f} via {row['source']} on {row['browser']}. "
            f"Age {row['age']}, sex {row['sex']}, country {row['country']}. "
            f"Account signed up {row['velocity_bucket']} before this purchase. "
            f"Purchase at hour {int(row['purchase_hour'])} "
            f"({'weekend' if row['is_weekend'] else 'weekday'}, "
            f"{'night' if row['is_night_purchase'] else 'day'}). "
            f"Device used by {int(row['device_user_count'])} distinct user(s). "
            f"Rolling spend over last 3 purchases: ${row['user_rolling_spend_3']:.2f}. "
            f"Rare country: {'yes' if row['is_rare_country'] else 'no'}."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_preprocess(self, use_cache=True):
        """
        Loads CSVs, runs all feature engineering stages, and caches the result.
        """
        cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
        if use_cache and os.path.exists(cache_file):
            print(f"Loading preprocessed data from cache: {cache_file}")
            self.df = pd.read_pickle(cache_file)
            return self.df

        print("Loading datasets...")
        df_fraud    = pd.read_csv(self.fraud_path)
        df_countries= pd.read_csv(self.ip_path)

        # Parse timestamps
        df_fraud['signup_time']   = pd.to_datetime(df_fraud['signup_time'])
        df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'])

        # --- Feature Engineering Pipeline ---
        print("  [1/6] Mapping IPs to countries...")
        df_fraud = self._map_ip_to_country(df_fraud, df_countries)

        print("  [2/6] Adding temporal features...")
        df_fraud = self._add_temporal_features(df_fraud)

        print("  [3/6] Adding velocity features...")
        df_fraud = self._add_velocity_features(df_fraud)

        print("  [4/6] Adding device-level features...")
        df_fraud = self._add_device_features(df_fraud)

        print("  [5/6] Adding user behavioral features...")
        df_fraud = self._add_user_behavioral_features(df_fraud)

        print("  [6/6] Adding country risk flag...")
        df_fraud = self._add_country_risk_feature(df_fraud)

        # Final chronological sort for RL sequential integrity
        self.df = df_fraud.sort_values(by='purchase_time').reset_index(drop=True)

        self.df.to_pickle(cache_file)
        print(f"Preprocessing complete. {len(self.df)} records cached.")
        return self.df

    def generate_embeddings(self, use_cache=True):
        """
        Builds a natural-language semantic string per transaction and encodes it
        with a SentenceTransformer, producing 384-dimensional dense vectors.
        """
        if self.df is None:
            raise ValueError("Run load_and_preprocess() before generating embeddings.")

        cache_file = os.path.join(self.cache_dir, "embeddings.npy")
        if use_cache and os.path.exists(cache_file):
            print(f"Loading embeddings from cache: {cache_file}")
            self.embeddings = np.load(cache_file)
            return self.embeddings

        print("Building semantic strings...")
        semantic_strings = self.df.apply(self._build_semantic_string, axis=1).tolist()

        print("Generating embeddings (this may take a while)...")
        self.embeddings = self.encoder.encode(
            semantic_strings,
            show_progress_bar=True,
            batch_size=256
        )

        np.save(cache_file, self.embeddings)
        print("Embeddings saved to cache.")
        return self.embeddings

    def get_train_test_split(self, train_ratio=0.8):
        """
        Chronological split to maintain causality for the MDP.
        Returns train/test dicts with embeddings, labels, and purchase values.
        """
        if self.embeddings is None:
            raise ValueError("Run generate_embeddings() before splitting.")

        split_idx = int(len(self.df) * train_ratio)

        labels = self.df['class'].values
        values = self.df['purchase_value'].values

        train_data = {
            'embeddings': self.embeddings[:split_idx],
            'labels':     labels[:split_idx],
            'values':     values[:split_idx],
        }
        test_data = {
            'embeddings': self.embeddings[split_idx:],
            'labels':     labels[split_idx:],
            'values':     values[split_idx:],
        }

        print(f"Chronological split: {len(train_data['labels'])} train / "
              f"{len(test_data['labels'])} test instances.")

        return train_data, test_data


if __name__ == "__main__":
    handler = FraudDataHandlerImproved(data_dir="./data")
    handler.load_and_preprocess()
    handler.generate_embeddings()
    train, test = handler.get_train_test_split()
