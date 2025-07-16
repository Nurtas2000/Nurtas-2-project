import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    """Load CSV data with error handling."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {path}")

def encode_categorical(df, column):
    """Encode categorical features."""
    le = LabelEncoder()
    df[f"{column}_encoded"] = le.fit_transform(df[column])
    return df, le

def save_plot(fig, filename):
    """Save matplotlib figure."""
    fig.savefig(f"{OUTPUT_PLOTS_DIR}/{filename}")
