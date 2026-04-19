from regress_model import *
from data_prep import *
import matplotlib.pyplot as plt
import numpy as np

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

# Trening na pełnym zbiorze uczącym
print("Rozpoczynanie finalnego treningu...")
for epoch in range(best_params['epochs']):
    for i in range(best_params['w_size'], 1600):
        x_in = scaled_data[i - best_params['w_size']:i]
        y_true = np.array([scaled_data[i]])
        final_model.train(x_in, y_true)


preds = []
reals = []
for i in range(1600, len(scaled_data)):
    x_in = scaled_data[i - best_params['w_size']:i]
    p, _ = final_model.forward(x_in)
    preds.append(denorm(p[0][0]))
    reals.append(denorm(scaled_data[i]))

residuals = np.array(reals) - np.array(preds)

# Wizualizacja wyników
plt.figure(figsize=(15, 7))
plt.plot(reals[:150], label='Dane rzeczywiste (Real)', color='#1f77b4', linewidth=2)
plt.plot(preds[:150], label='Predykcja modelu (RNN)', color='#ff7f0e', linestyle='--', linewidth=2)


plt.title(f"Finalna Prognoza Temperatury (MAE: 0.5009)", fontsize=14)
plt.xlabel("Kolejne kroki czasowe", fontsize=12)
plt.ylabel("Temperatura [°C]", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('finalny_wykres_sukcesu.png')
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(reals, preds, alpha=0.5, color='purple')
plt.plot([min(reals), max(reals)], [min(reals), max(reals)], color='red', lw=2)
plt.title("Zależność: Predykcja vs Wartość Rzeczywista")
plt.xlabel("Wartość rzeczywista [°C]")
plt.ylabel("Predykcja modelu [°C]")
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(residuals[:150], color='red', label='Błąd (Real - Pred)')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Wykres błędów modelu w czasie (Residuals)", fontsize=14)
plt.xlabel("Krok czasowy", fontsize=12)
plt.ylabel("Różnica [°C]", fontsize=12)
plt.fill_between(range(len(residuals[:150])), residuals[:150], alpha=0.2, color='red')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='green', edgecolor='black', alpha=0.7)
plt.title("Rozkład błędów modelu")
plt.xlabel("Błąd [°C]")
plt.ylabel("Częstotliwość")
plt.show()

