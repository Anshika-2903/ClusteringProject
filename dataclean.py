# dataclean.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Example cleaning steps
    df = df.dropna()  # Remove missing values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=["float64", "int64"]))
    return scaled_data
