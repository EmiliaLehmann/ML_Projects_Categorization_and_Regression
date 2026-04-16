from regress_model import *
from data_prep import *
import matplotlib.pyplot as plt

best_params = {
    'w_size': 18,
    'h_size': 32,
    'layers': 3,
    'lr': 0.001,
    'act': 'elu',
    'epochs': 30
}

final_model = TemperaturePredictionNet(
    input_size=1,
    hidden_size=best_params['h_size'],
    output_size=1,
    lr=best_params['lr'],
    num_layers=best_params['layers'],
    activation=best_params['act']
)


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


plt.figure(figsize=(15, 7))
plt.plot(reals[:150], label='Dane rzeczywiste (Real)', color='#1f77b4', linewidth=2)
plt.plot(preds[:150], label='Predykcja modelu (RNN)', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title(f"Finalna Prognoza Temperatury (MAE: {0.5354})", fontsize=14)
plt.xlabel("Kolejne kroki czasowe", fontsize=12)
plt.ylabel("Temperatura [°C]", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('finalny_wykres_sukcesu.png')
plt.show()

print("Wykres został zapisany jako 'finalny_wykres_sukcesu.png'")