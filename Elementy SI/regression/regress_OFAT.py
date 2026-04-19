from regress_model import *
from data_prep import *


def save_results_to_files(param_name, val, avg_train, avg_test):
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # 1. Zapis do CSV
    csv_file = 'wyniki_analizy_rnn.csv'
    df_entry = pd.DataFrame([{
        'Data': timestamp,
        'Analizowany_Parametr': param_name,
        'Wartosc': val,
        'MAE_Train': round(avg_train, 4),
        'MAE_Test': round(avg_test, 4),
        'Roznica_Overfitting': round(abs(avg_train - avg_test), 4)
    }])
    df_entry.to_csv(csv_file, mode='a', index=False, header=not pd.io.common.file_exists(csv_file))

    # 2. Zapis do TXT
    with open("raport_koncowy.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] PARAMETR: {param_name:15} | WARTOŚĆ: {str(val):8}\n")
        f.write(f"  -> MAE (Ucząca): {avg_train:.4f}\n")
        f.write(f"  -> MAE (Testowa): {avg_test:.4f}\n")
        f.write("-" * 50 + "\n")


# LISTA PARAMETRÓW DO ANALIZY
analysis_config = {
    "num_layers": [1, 2, 3, 4],
    "activation": ['tanh', 'relu', 'sigmoid', 'elu'],
    "window_size": [6, 12, 18, 24],
    "hidden_size": [16, 32, 64, 128],
    "learning_rate": [0.001, 0.005, 0.01, 0.05],
    "training_size": [400, 800, 1200, 1600],
    "epochs": [25, 50, 75, 100]
}

base_params = {
    'w_size': 12,
    'h_size': 32,
    'lr': 0.005,
    'tr_size': 1200,
    'layers': 1,
    'act': 'tanh',
    'epochs': 50
}

n_repeats = 10  # epowtórzenie procesu kilkakrotnie

print("Boze daj mi sile zaczynamy walke tak nierowna jak atomic baby vs hydrogen bomb")

for param_name, values in analysis_config.items():
    print(f"\n>>> Badanie wpływu: {param_name}")

    for val in values:
        current_run = base_params.copy()

        # MAPOWANIE PARAMETRÓW
        if param_name == "num_layers":
            current_run['layers'] = val
        elif param_name == "activation":
            current_run['act'] = val
        elif param_name == "window_size":
            current_run['w_size'] = val
        elif param_name == "hidden_size":
            current_run['h_size'] = val
        elif param_name == "learning_rate":
            current_run['lr'] = val
        elif param_name == "training_size":
            current_run['tr_size'] = val
        elif param_name == "epochs":
            current_run['epochs'] = val

        repeat_train_mae = []
        repeat_test_mae = []

        for r in range(n_repeats):
            model = TemperaturePredictionNet(
                input_size=1,
                hidden_size=current_run['h_size'],
                output_size=1,
                lr=current_run['lr'],
                num_layers=current_run['layers'],
                activation=current_run['act']
            )

            # TRENING
            epoch_train_errors = []
            for epoch in range(current_run['epochs']):
                for i in range(current_run['w_size'], current_run['tr_size']):
                    x_in = scaled_data[i - current_run['w_size']:i]
                    y_true = np.array([scaled_data[i]])
                    model.train(x_in, y_true)
                    pred, _ = model.forward(x_in)
                    err = np.abs(denorm(y_true[0]) - denorm(pred[0][0]))
                    epoch_train_errors.append(err)

            # TEST
            test_errors = []
            for i in range(current_run['tr_size'], len(scaled_data)):
                x_in = scaled_data[i - current_run['w_size']:i]
                pred, _ = model.forward(x_in)
                err = np.abs(denorm(scaled_data[i]) - denorm(pred[0][0]))
                test_errors.append(err)

            repeat_train_mae.append(np.mean(epoch_train_errors))
            repeat_test_mae.append(np.mean(test_errors))

        final_avg_train = np.mean(repeat_train_mae)
        final_avg_test = np.mean(repeat_test_mae)

        # Zapis do plików
        save_results_to_files(param_name, val, final_avg_train, final_avg_test)
        print(f"  Konfiguracja {val} zakończona. Test MAE: {final_avg_test:.4f}")

print("\ntesty zakonczone")