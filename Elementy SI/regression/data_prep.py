import numpy as np
import pandas as pd

# 1. Wczytywanie i czyszczenie
df = pd.read_csv("dataset_reg.csv")
data = df['LandAverageTemperature'].dropna().values

# 2. Skalowanie danych do zakresu [-1, 1]
data_min, data_max = np.min(data), np.max(data)
scaled_data = (data - data_min) / (data_max - data_min) * 2 - 1

# 3. Funkcje pomocnicze
def create_windows(dataset, window_size=12):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i + window_size])
        y.append(dataset[i + window_size])
    return np.array(X).reshape(-1, window_size, 1), np.array(y).reshape(-1, 1)

def denorm(x):
    return ((x + 1) / 2) * (data_max - data_min) + data_min