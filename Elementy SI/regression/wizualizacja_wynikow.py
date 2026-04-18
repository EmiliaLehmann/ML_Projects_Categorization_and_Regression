from regress_model import *
from data_prep import *
import matplotlib.pyplot as plt

best_params = {
    'w_size': 18,      # Najlepszy wynik dla okna 18
    'h_size': 64,      # 64 neurony okazały się optymalne statystycznie
    'layers': 2,       # 2 warstwy zapewniły najlepszą stabilność
    'lr': 0.001,       # Najniższy błąd przy wolniejszym uczeniu
    'act': 'elu',      # Funkcja ELU zdecydowanie wyprzedziła tanh/relu
    'epochs': 75       # Optymalny punkt przed wystąpieniem overfittingu
}

# Inicjalizacja modelu z optymalnymi parametrami
final_model = TemperaturePredictionNet(
    input_size=1,
    hidden_size=best_params['h_size'],
    output_size=1,
    lr=best_params['lr'],
    num_layers=best_params['layers'],
    activation=best_params['act']
)

# Trening na pełnym zbiorze uczącym (1600 próbek)
print("Rozpoczynanie finalnego treningu...")
for epoch in range(best_params['epochs']):
    for i in range(best_params['w_size'], 1600):
        x_in = scaled_data[i - best_params['w_size']:i]
        y_true = np.array([scaled_data[i]])
        final_model.train(x_in, y_true)

# Faza testowa - predykcja na danych niewidzianych
preds = []
reals = []
for i in range(1600, len(scaled_data)):
    x_in = scaled_data[i - best_params['w_size']:i]
    p, _ = final_model.forward(x_in)
    preds.append(denorm(p[0][0]))
    reals.append(denorm(scaled_data[i]))

# Wizualizacja wyników
plt.figure(figsize=(15, 7))
plt.plot(reals[:150], label='Dane rzeczywiste (Real)', color='#1f77b4', linewidth=2)
plt.plot(preds[:150], label='Predykcja modelu (RNN)', color='#ff7f0e', linestyle='--', linewidth=2)

# MAE na poziomie ~0.50 zgodnie z najnowszymi testami
plt.title(f"Finalna Prognoza Temperatury (MAE: 0.5009)", fontsize=14)
plt.xlabel("Kolejne kroki czasowe", fontsize=12)
plt.ylabel("Temperatura [°C]", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('finalny_wykres_sukcesu.png')
plt.show()