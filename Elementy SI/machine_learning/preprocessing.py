import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def simplify_genre_detailed(genre):
    genre = str(genre).lower()

    if 'classical' in genre or 'opera' in genre: return 'Classical'
    if 'jazz' in genre: return 'Jazz'
    if 'blues' in genre or 'bluegrass' in genre: return 'Blues'

    if any(word in genre for word in ['metal', 'grindcore', 'hardcore']): return 'Metal'
    if any(word in genre for word in ['punk', 'grunge']): return 'Punk/Grunge'
    if 'rock' in genre or 'alternative' in genre or 'emo' in genre: return 'Rock'

    if any(word in genre for word in
           ['techno', 'trance', 'dubstep', 'hardstyle', 'drum-and-bass']): return 'Hard-Electronic'
    if any(word in genre for word in ['house', 'edm', 'electro', 'idm', 'breakbeat']): return 'Dance-Electronic'
    if 'ambient' in genre or 'new-age' in genre or 'sleep' in genre: return 'Ambient/Sleep'

    if 'indie' in genre: return 'Indie'
    if 'pop' in genre: return 'Pop'

    if 'hip-hop' in genre or 'rap' in genre: return 'Hip-Hop'
    if any(word in genre for word in ['r-n-b', 'soul', 'funk', 'groove']): return 'RnB/Soul'
    if 'reggae' in genre or 'ska' in genre or 'dub' in genre: return 'Reggae/Ska'
    if 'reggaeton' in genre: return 'Reggaeton'

    if any(word in genre for word in ['latin', 'latino', 'salsa', 'samba', 'tango', 'spanish']): return 'Latin'
    if any(word in genre for word in ['brazil', 'mpb', 'pagode', 'forro', 'sertanejo']): return 'Brazil-Regional'
    if any(word in genre for word in
           ['afrobeat', 'world-music', 'indian', 'iranian', 'malay', 'turkish']): return 'World'

    if 'country' in genre or 'honky-tonk' in genre: return 'Country'
    if 'folk' in genre or 'songwriter' in genre: return 'Folk'
    if 'acoustic' in genre or 'piano' in genre or 'guitar' in genre: return 'Acoustic/Instrumental'

    if 'kids' in genre or 'children' in genre or 'disney' in genre or 'anime' in genre: return 'Kids'
    if any(word in genre for word in ['party', 'club', 'disco', 'happy']): return 'Party/Disco'
    if any(word in genre for word in ['chill', 'study', 'sad', 'romance']): return 'Chill/Mood'

    return 'Other'
#biore sobie uproszczenie tych nazw z funkcji wcześniejszej <3

def load_classification_data(
    file_path="../classification/dataset_cat.csv",
    test_size=0.2,
    random_state=42
):
    df = pd.read_csv(file_path)
    df = df.dropna()

    if 'track_name' in df.columns and 'artists' in df.columns:
        df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')
    else:
        df = df.drop_duplicates()
    #pozbywamy się duplikatów

    df['track_genre'] = df['track_genre'].apply(simplify_genre_detailed)
    #upraszczamy nazwy gatunków

    selected_features = [
        'popularity', 'duration_ms', 'danceability', 'energy',
        'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'explicit', 'key', 'mode',
        'time_signature'
    ]

    X = df[selected_features].copy()
    y = df['track_genre'].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    #zamiana klas na liczby

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    #podział, 20% dla testowego zbioru

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #skalowanie wartości

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": selected_features,
        "class_names": label_encoder.classes_,
        "label_encoder": label_encoder,
        "scaler": scaler
    }


def create_regression_windows(series, window_size=12):
    X, y = [], []

    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])

    return np.array(X), np.array(y)
#tworzymy okna po 12 miesięcy, i bedziemy przewidywac miesiąc 13

def load_regression_data(
    file_path="../regression/dataset_reg.csv",
    window_size=12,
    test_size=0.2
):
    df = pd.read_csv(file_path)

    data = df['LandAverageTemperature'].dropna().values.astype(float)
    #biorę temperatury, usuwam brakujące wartości, i daje na float

    data_min = np.min(data)
    data_max = np.max(data)

    scaled_data = (data - data_min) / (data_max - data_min)
    #skaluje do przedziału 0-1

    X, y = create_regression_windows(scaled_data, window_size=window_size)
    #tworze okna, X przeszlosc 12 miesiecy, Y dla miesiaca 13

    split_index = int(len(X) * (1 - test_size))
    #punkt podziału dla danych

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    #i podział ich
    #nie używamy train_test_split bo chcemy zachowac odpowiednie dane, do train muszą byc przeszle, a do test przyszle

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "window_size": window_size,
        "data_min": data_min,
        "data_max": data_max
    }


def inverse_scale_temperature(values, data_min, data_max):
    values = np.array(values)
    return values * (data_max - data_min) + data_min
#do odwrócenia skalowania wartości


# if __name__ == "__main__":
#     data = load_classification_data()
#
#     print("X_train shape:", data["X_train"].shape)
#     print("X_test shape:", data["X_test"].shape)
#     print("y_train shape:", data["y_train"].shape)
#     print("y_test shape:", data["y_test"].shape)
#
#     print("\nPrzykładowe klasy:", data["class_names"][:5])


