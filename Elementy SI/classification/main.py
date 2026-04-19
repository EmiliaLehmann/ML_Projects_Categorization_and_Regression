import matplotlib.pyplot as plt
import numpy as np
from data_cleaning import prepare_data
from model_class import DeepSpotifyNet
from train import train_model

# Przygotowanie danych
X_train, X_test, y_train_oh, y_test_oh, y_test, num_classes, le = prepare_data("dataset_cat.csv")

# Konfiguracja modelu
input_dim = X_train.shape[1]
epochs = 150
hidden_size1=1024
hidden_size2=512
hidden_size3=256
hidden_size4=128
activations_to_test='relu'
learning_rate = 0.001
batch_size =256
dropout_rate=0.2


###################################################################################################################
print(f"\n>>> Test: LR={learning_rate}, Batch={batch_size}, Hidden={hidden_size1}, Act={activations_to_test}")

nn = DeepSpotifyNet(
     input_size=input_dim,
     hidden_size1=1024,
      hidden_size2=512,
     hidden_size3=256,
     # hidden_size4=  128,
     output_size=num_classes,
     lr=learning_rate,
     activation=activations_to_test
)

 #  Trenowanie
print("\nRozpoczynam trenowanie sieci...")
nn = train_model(nn, X_train, y_train_oh, epochs=epochs, batch_size=batch_size)

# Ewaluacja
test_probs = nn.forward(X_test, training=False)
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
    f.write(f"Hidden size2: {hidden_size2}\n")
    f.write(f"Hidden size3: {hidden_size3}\n")
    # f.write(f"Hidden size4: {hidden_size4}\n")
    f.write(f"Wynik dla: {activations_to_test}\n")
    f.write(f"Skutecznosc (Accuracy): {accuracy * 100:.2f}%\n")
    f.write("----------------------------------------\n")

##############################################################################



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
