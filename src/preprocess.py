from pathlib import Path
import pandas as pd
FEATURES=[f"q{i}" for i in range(1,11)]
TARGET="profile"
def load_dataset(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[1]/"data"/"geek_dataset.csv"
    return pd.read_csv(path)
def split_features_target(df):
    return df[FEATURES].astype(int),df[TARGET]
