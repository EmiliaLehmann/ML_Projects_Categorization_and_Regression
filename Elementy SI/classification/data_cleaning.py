import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from dython.nominal import associations


def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    if 'track_name' in df.columns and 'artists' in df.columns:
        df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')
    else:
        df = df.drop_duplicates()

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

    df['track_genre'] = df['track_genre'].apply(simplify_genre_detailed)

    #
    # top_genres = df['track_genre'].value_counts().nlargest(10).index
    # df = df[df['track_genre'].isin(top_genres)]
    # print(f"Liczba gatunków: {len(genres)}")
    # print(genres)



    # Label Encoding dla targetu
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['track_genre'])

    # Selekcja cech numerycznych (ANOVA)
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Unnamed: 0' in num_features: num_features.remove('Unnamed: 0')

    f_values, _ = f_classif(df[num_features], y_encoded)
    f_values_norm = (f_values - f_values.min()) / (f_values.max() - f_values.min())
    corr_num = pd.Series(f_values_norm, index=num_features)

    categorical_cols = []

    selected_features = ['popularity', 'duration_ms', 'danceability', 'energy',
                         'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                         'liveness', 'valence', 'tempo', 'explicit', 'key', 'mode','time_signature' ]



    X = df[selected_features]
    y = y_encoded

    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Skalowanie
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One-hot encoding dla targetu (do liczenia straty w sieci)
    num_classes = len(le.classes_)
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh = np.eye(num_classes)[y_test]

    return X_train_scaled, X_test_scaled, y_train_oh, y_test_oh, y_test, num_classes, le


