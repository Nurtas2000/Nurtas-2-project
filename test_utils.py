import pytest
from src.utils import load_data, encode_categorical

def test_load_data():
    df = load_data("../../data/fitness_data.csv")
    assert df.shape == (50, 7)

def test_encode_categorical():
    df = pd.DataFrame({"Activity": ["Running", "Cycling"]})
    df_encoded, _ = encode_categorical(df, "Activity")
    assert "Activity_encoded" in df_encoded.columns
