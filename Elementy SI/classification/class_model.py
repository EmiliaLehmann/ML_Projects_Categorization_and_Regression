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
selected_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'key',
    'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature', 'explicit']
X = df[selected_features]

y = y_encoded

#podzial na dane do testowania i trenowania modelu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

# dopasowanie i transformacja zbioru treningowego
X_train_scaled = scaler.fit_transform(X_train)

# transfromacja zbioru testowego
X_test_scaled = scaler.transform(X_test)

#tu zamiast 113 roznych gatunkow mamy macierz rip
num_classes = len(le.classes_)
y_train_oh = np.eye(num_classes)[y_train]
y_test_oh = np.eye(num_classes)[y_test]

#no to teraz musze napisac model xDDD
#to jest sklejka z dykty, internetu, moich marzen i poprawek gemini ok ale mamy tu ReLL takze super chcialam tego uzyc
#brat ma na razie

class DeepSpotifyNet:
    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3, output_size, lr=0.01):
        self.lr = lr
        # Inicjalizacja wag He - ustawiamy wagi losowo oraz biasy tak, zeby neurony dzialaly nawet ze slabym sygnalem na wejsciu
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size1))

        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))

        self.W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((1, hidden_size3))

        self.W4 = np.random.randn(hidden_size3, output_size) * np.sqrt(2. / hidden_size3)
        self.b4 = np.zeros((1, output_size))
        self.loss_history = []

    #tu jest nasz kochany rell - wartosci < 0 = 0 i te wyzej leca liniowo
    def _relu(self, x):
        return np.maximum(0, x)

    #softmax zamienia liczby na prawdopodobienstwa
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    #mnozenie macierzy here -.dot,  uzycie relu oraz softmaxa, dodajemy tez biasy (bangchan??)
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._relu(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self._relu(self.z3)

        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.probs = self._softmax(self.z4)

        self.probs = self._softmax(self.z2)
        return self.probs

    def train(self, X, y_oh, epochs=50, batch_size=128):
        m = X.shape[0]
        for epoch in range(epochs):
            # Mieszanie danych w każdej epoce
            # mieszamy dane dbajac o roznorodne przyklady a nie 1000 piosenek tego samego gatunku pod rzad
            perm = np.random.permutation(m)
            X_shuffled = X[perm]
            y_shuffled = y_oh[perm]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward
                probs = self.forward(X_batch)

                # Backward
                # Sieć patrzy na swoje przewidywanie (probs) i porównuje je z prawdą (y_batch).
                # dz2 = probs - y_batch: To jest nasz błąd. Mówi nam, o ile sieć się pomyliła.

                dz4 = probs - y_batch
                dW4 = np.dot(self.a3.T, dz4) / batch_size
                db4 = np.sum(dz4, axis=0, keepdims=True) / batch_size

                dz3 = np.dot(dz4, self.W4.T) * (self.z3 > 0)
                dW3 = np.dot(self.a2.T, dz3) / batch_size
                db3 = np.sum(dz3, axis=0, keepdims=True) / batch_size

                dz2 = np.dot(dz3, self.W3.T) * (self.z2 > 0)

                dW2 = np.dot(self.a1.T, dz2) / batch_size
                db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

                # Sieć sprawdza:
                # Które wagi w ostatniej warstwie (dW2) najbardziej przyczyniły się do błędu?
                # Idąc głębiej: które wagi w pierwszej warstwie (dW1) wysłały złe sygnały?

                dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0)
                dW1 = np.dot(X_batch.T, dz1) / batch_size
                db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

                # Update
                # poprawiamy zle wagi (te ktore nie prowadza do zachcianych przez nas efektow)
                # Używamy lr (Learning Rate), żeby nie zmieniać wag zbyt gwałtownie
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

                self.W3 -= self.lr * dW3
                self.b3 -= self.lr * db3
                self.W4 -= self.lr * dW4
                self.b4 -= self.lr * db4

            # Obliczanie straty na koniec epoki - jak bardzo brat sie pomylil
            full_probs = self.forward(X)
            #entropia krzyzowa
            loss = -np.mean(np.sum(y_oh * np.log(full_probs + 1e-10), axis=1))
            #sledzenie loss history zeby zobaczyc czy model sie uczy
            self.loss_history.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# odpalamy kolege oh god oh fuck

#  Inicjalizacja (8 wejść, 64 neurony ukryte, 114 gatunków)
input_dim = X_train_scaled.shape[1]
output_dim = num_classes
nn = DeepSpotifyNet(input_size=input_dim, hidden_size1=512,hidden_size2=256, hidden_size3=128, output_size=output_dim, lr=0.05)

# Trenowanie
print("\nRozpoczynam trenowanie sieci...")
nn.train(X_train_scaled, y_train_oh, epochs=200, batch_size=256)

#  Wykres Loss - czy siec sie uczy?
plt.plot(nn.loss_history)
plt.title("Postęp uczenia się (Loss)")
plt.xlabel("Epoka")
plt.ylabel("Błąd (Cross-Entropy)")
plt.show()

# skutecznosc modelu
test_probs = nn.forward(X_test_scaled)
predictions = np.argmax(test_probs, axis=1)
accuracy = np.mean(predictions == y_test)
print(f"\nSkuteczność na zbiorze testowym: {accuracy * 100:.2f}%")

# Zapisywanie wyniku do pliku tekstowego
with open("wynik_modelu.txt", "a") as f:
    f.write("\n" + "="*40 + "\n")
    f.write("--- NOWY TEST MODELU ---\n")
    f.write(f"Liczba epok: \n")
    f.write(f"Learning Rate: \n")
    f.write(f"Skutecznosc (Accuracy): {accuracy * 100:.2f}%\n")
    f.write("Zmiany:\n")
    f.write("----------------------------------------\n")


print("\nWynik został dopisany do pliku wynik_modelu.txt")