import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    # Drop columns with too many missing values
    df = df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"], errors='ignore')

    # Fill numeric missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical missing values
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # Encode categorical
    df = pd.get_dummies(df, drop_first=True)

    return df