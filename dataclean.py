import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath, delimiter=';'):
    # Load data
    data = pd.read_csv(filepath, delimiter=delimiter)

    # Drop duplicates
    data = data.drop_duplicates()

    # Strip column names
    data.columns = data.columns.str.strip()

    # Replace missing values with the mean of each column
    data['Latitude'].fillna(data['Latitude'].mean(), inplace=True)
    data['Longitude'].fillna(data['Longitude'].mean(), inplace=True)

    # Reset index
    data = data.reset_index(drop=True)

    return data

def scale_data(data):
    # Initialize the scaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled
