import matplotlib.pyplot as plt
import numpy as np
from data_cleaning import prepare_data
from model_class import DeepSpotifyNet
from train import train_model

# Przygotowanie danych
X_train, X_test, y_train_oh, y_test_oh, y_test, num_classes, le = prepare_data("dataset_cat.csv")

# Konfiguracja modelu
input_dim = X_train.shape[1]
epochs = 100


learning_rate = 0.05
    #[0.1, 0.5, 0.01, 0.05]
hidden_size1=512
  #  [128,256,512,1024]
batch_size=64
    #[64,128,256,512]
activations_to_test ='relu'
  #  ['relu', 'sigmoid', 'tanh', 'leaky_relu']

results = []
# for lea in learning_rate:
#     for bat in batch_size:
#         for hid in hidden_size1:
#             for act in activations_to_test:
#                 print(f"\n>>> Test: LR={lea}, Batch={bat}, Hidden={hid}, Act={act}")
#
#
#
#                 nn = DeepSpotifyNet(
#                     input_size=input_dim,
#                     hidden_size1=hid,
#                     output_size=num_classes,
#                     lr=lea,
#                     activation = act
#                 )
#
#
#                 print("\nRozpoczynam trenowanie sieci...")
#                 nn = train_model(nn, X_train, y_train_oh, epochs=epochs, batch_size=bat)
#
#
#                 test_probs = nn.forward(X_test)
#                 predictions = np.argmax(test_probs, axis=1)
#                 accuracy = np.mean(predictions == y_test)
#
#                # print(f"Wynik tej próby: {accuracy * 100:.2f}%")
#
#
#
#
#                 with open("wynik_modelu.txt", "a") as f:
#                     f.write("\n" + "="*40 + "\n")
#                     f.write("--- NOWY TEST MODELU ---\n")
#                     f.write(f"Liczba epok: {epochs}\n")
#                     f.write(f"Learning Rate: {learning_rate}\n")
#                     f.write(f"Wynik dla: {act}")
#                     f.write(f"Skutecznosc (Accuracy): {accuracy * 100:.2f}%\n")
#                     f.write("----------------------------------------\n")
#
#                 # print("\nWynik został dopisany do pliku wynik_modelu.txt")
#                 results.append({
#                     'accuracy': accuracy,
#                     'activations': act,
#                     'hidden': hid,
#                     'lr': lea,
#                     'batch': bat,
#
#                 })

# print("\n" + "="*50)
#
# best_result = max(results, key=lambda x: x['accuracy'])
# with open("wynik_modelu.txt", "a") as f:
#     f.write("!!! NAJLEPSZY OSIĄGNIĘTY WYNIK !!!")
#     f.write(f"Accuracy: {best_result['accuracy'] * 100:.2f}%")
#     f.write(f"Parametry:")
#     f.write(f" - Funkcja aktywacji: {best_result['activation']}")
#     f.write(f" - Rozmiar warstwy:    {best_result['hidden']}")
#     f.write(f" - Learning Rate:      {best_result['lr']}")
#     f.write(f" - Batch Size:         {best_result['batch']}")
#     f.write("="*50)

print(f"\n>>> Test: LR={learning_rate}, Batch={batch_size}, Hidden={hidden_size1}, Act={activations_to_test}")

nn = DeepSpotifyNet(
    input_size=input_dim,
    hidden_size1=hidden_size1,
    hidden_size2=256,
    output_size=num_classes,
    lr=learning_rate,
    activation=activations_to_test
)

#  Trenowanie
print("\nRozpoczynam trenowanie sieci...")
nn = train_model(nn, X_train, y_train_oh, epochs=epochs, batch_size=batch_size)

# Ewaluacja
test_probs = nn.forward(X_test)
predictions = np.argmax(test_probs, axis=1)
accuracy = np.mean(predictions == y_test)

# Zapisywanie wyniku
with open("wynik_modelu.txt", "a") as f:
    f.write("\n" + "=" * 40 + "\n")
    f.write("--- NOWY TEST MODELU ---\n")
    f.write(f"Liczba epok: {epochs}\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Hidden size: {hidden_size1}\n")
    f.write(f"Wynik dla: {activations_to_test}\n")
    f.write(f"Skutecznosc (Accuracy): {accuracy * 100:.2f}%\n")
    f.write("----------------------------------------\n")

