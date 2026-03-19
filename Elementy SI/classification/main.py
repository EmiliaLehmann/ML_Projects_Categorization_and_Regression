import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
from dython.nominal import associations
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Download latest version
#path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")

df = pd.read_csv("dataset.csv")

#czyszczenie danych i przygotowanie pod model
print(df.info())
df = df.dropna()

#przypisanie slownym wartosciom naszego targetu - genre - labelu
le = LabelEncoder()
y_encoded = le.fit_transform(df['track_genre'])

#wybranie cech numerycznych do modelu i policzenie ich korelacji z genre - targetem
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Unnamed: 0' in num_features: num_features.remove('Unnamed: 0')

#policzenie F-score bo to z testu ANOVA jest + ich normalizacja do 0-1
f_values, _ = f_classif(df[num_features], y_encoded)
f_values_norm = (f_values - f_values.min()) / (f_values.max() - f_values.min())
corr_num = pd.Series(f_values_norm, index=num_features)

# korelacja > 0.4 to prog do wyboru feature do trenowania modelu
selected_num_features = corr_num[corr_num > 0.4].index.tolist()

print("cechy numeryczne skorelowane w >0,4 stopniu:", selected_num_features)

#wybranie cech kategorycznych do modelu (tu mamy tylko jedną chyba)
cat_features = ['explicit', 'track_genre']

# korelacja genre i jego bycia explicit
assoc = associations(df[cat_features], nominal_columns='all', plot=False)
corr_matrix = assoc['corr']

# bierzemy mniejszy prog 0,3 do znowu zobaczenia czy explicit to znaczacy parametr
selected_cat_features = corr_matrix[corr_matrix['track_genre'] > 0.3].index.tolist()
selected_cat_features.remove('track_genre')

print("Czy explicit jest jakkolwiek wartosciowa zmienna? Jak sie tu pojawi, to tak", selected_cat_features)

#target
y = df['track_genre']

#parametry
X = df['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'explicit']

#podzial na dane do testowania i trenowania modelu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

# dopasowanie i transformacja zbioru treningowego
X_train_scaled = scaler.fit_transform(X_train)

# transfromacja zbioru testowego
X_test_scaled = scaler.transform(X_test)

#no to teraz musze napisac model xDDD

