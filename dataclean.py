# data_clean.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, delimiter=';')

    # Data Cleaning
    data = data.drop_duplicates()
    data.columns = data.columns.str.strip()
    data['Latitude'].fillna(data['Latitude'].mean(), inplace=True)
    data['Longitude'].fillna(data['Longitude'].mean(), inplace=True)
    data = data.reset_index(drop=True)

    # Data Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data, data_scaled
