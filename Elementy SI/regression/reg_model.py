import numpy as np
import pandas as pd
import kagglehub

# Download latest version
# path = kagglehub.dataset_download("berkeleyearth/climate-change-earth-surface-temperature-data")

df = pd.read_csv("dataset_reg.csv")

data = df['LandAverageTemperature'].dropna().values

# skalujemy dane o temperaturze do wartosci [-1,1]
data_min = np.min(data)
data_max = np.max(data)
scaled_data = (data - data_min) / (data_max - data_min) * 2 - 1

#robimy przedziały (okna) do ktorych wpadają dane, tu jest 12 miesięcy pod rząd branych
def create_windows(dataset, window_size=12):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i + window_size])
        y.append(dataset[i + window_size])
    return np.array(X).reshape(-1, window_size, 1), np.array(y).reshape(-1, 1)

#przygotowujemy dane do wsadzenia do modelu przez wsadzenie je do tych okien wczesniej wspomnianych
X_train, y_train = create_windows(scaled_data)

#to jest nasz model głowny jak w tym klasyfikacyjnym
class TemperaturePredictionNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.hidden_size = hidden_size
        self.lr = lr

        # ustawiamy randomowe wagi na poczatku
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(output_size, hidden_size) * 0.1
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, x):
        h = {-1: np.zeros((self.hidden_size, 1))}
        for t in range(len(x)):
            # to chyba jest odwektorowanie tego xt
            xt = x[t].reshape(-1, 1)
            # to jest jakas taka funkcja especial for szeregi czasowe h_t = tanh(Wxh * xt + Whh * h_{t-1} + bh)
            h[t] = np.tanh(np.dot(self.Wxh, xt) + np.dot(self.Whh, h[t - 1]) + self.bh)

        y_pred = np.dot(self.Why, h[len(x) - 1]) + self.by
        return y_pred, h

    #tu jest wsm prawie to samo co w tym klasyfikacyjnym
    def train(self, x, y_true):
        y_pred, h = self.forward(x)
        dy = y_pred - y_true.reshape(-1, 1)  # tu jest ten blad do kwadratu co w excelu liczylismy

        dWhy = np.dot(dy, h[len(x) - 1].T)
        dby = dy

        # pierwszy hidden gradient ktory bedziemy poprawiac
        dh = np.dot(self.Why.T, dy)

        dWxh, dWhh, dbh = 0, 0, 0

        # propagacja wsteczna wg czasu - tu miesiecy
        for t in reversed(range(len(x))):
            # wsadzamu do tej specjalnej funkcji nasz gradient
            dtanh = (1 - h[t] ** 2) * dh

            dbh += dtanh
            dWxh += np.dot(dtanh, x[t].reshape(-1, 1).T)
            dWhh += np.dot(dtanh, h[t - 1].T)

            # ten gradient wciagany jest do tego hidden state
            dh = np.dot(self.Whh.T, dtanh)

        for param, grad in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                               [dWxh, dWhh, dWhy, dbh, dby]):
            # tu gemini said ze to ma byc bo mi inaczej brat nie dziala i gradient ma byc ograniczony zeby mi
            #sie wszystkie wartosci nie wyzerowały???
            np.clip(grad, -5, 5, out=grad)
            param -= self.lr * grad

        return 0.5 * (y_pred - y_true) ** 2  # znowu ta strata z excela kwadraty bledu


#zapis danych z prób
def save_results_to_files(window, hidden, lr, mae_result):
    # dane do zapisu
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # zapis csv
    csv_file = 'eksperymenty_rnn.csv'
    df_entry = pd.DataFrame([{
        'Data': timestamp,
        'Window_Size': window,
        'Hidden_Size': hidden,
        'Learning_Rate': lr,
        'MAE_Celsius': round(mae_result, 4)
    }])

    # struktura plikow csv
    df_entry.to_csv(csv_file, mode='a', index=False, header=not pd.io.common.file_exists(csv_file))

    # zapis do txt
    with open("logi_modelu.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Sesja: {timestamp} ---\n")
        f.write(f"Parametry: okno={window}, ukryte={hidden}, lr={lr}\n")
        f.write(f"Wynik MAE: {mae_result:.4f} °C\n")
        f.write("-" * 30 + "\n")

#to jest inny typ doboru danych (walk forward validation) - nie random wybierane jak w naszym modelu z kategoryzacja tylko duzo na raz po sobie
#no bo ma przewidywac w czasie po sobie a nie random ahh dane z 2021 a potem 1939

#no tu w sumie parametry do modelu dobieramy
windows_to_test = [6, 12, 24]
hiddens_to_test = [32, 64]
lrs_to_test = [0.001, 0.005]
train_size = 1200

results_summary = []

print("Grid Search start")

for w_size in windows_to_test:
    for h_size in hiddens_to_test:
        for lr_rate in lrs_to_test:

            print(f"\n>>> Testowanie: Window={w_size}, Hidden={h_size}, LR={lr_rate}")

            # Inicjalizacja modelu z konkretnymi parametrami z tej iteracji
            model = TemperaturePredictionNet(input_size=1, hidden_size=h_size, output_size=1, lr=lr_rate)

            wf_predictions = []
            wf_actuals = []

            for i in range(train_size, len(scaled_data)):
                x_input = scaled_data[i - w_size:i].reshape(w_size, 1)
                y_true = scaled_data[i]

                y_pred, _ = model.forward(x_input)
                wf_predictions.append(y_pred[0][0])
                wf_actuals.append(y_true)

                model.train(x_input, np.array([y_true]))

            #liczenie po ludzku wartosci w celicjluszach
            def denorm(x):
                return ((x + 1) / 2) * (data_max - data_min) + data_min

            p = denorm(np.array(wf_predictions))
            a = denorm(np.array(wf_actuals))
            current_mae = np.mean(np.abs(p - a))

            print(f"Koniec iteracji. MAE: {current_mae:.4f} °C")

            # Zapisujemy konkretne wartości z TEJ iteracji (w_size, h_size, lr_rate)
            save_results_to_files(w_size, h_size, lr_rate, current_mae)

            results_summary.append({
                'window': w_size,
                'hidden': h_size,
                'lr': lr_rate,
                'mae': current_mae
            })

print("\n" + "="*30)
print("GRID SEARCH ZAKOŃCZONY!")

# Wyświetlenie rankingu
df_results = pd.DataFrame(results_summary)
best_result = df_results.loc[df_results['mae'].idxmin()]

print("\n--- NAJLEPSZE PARAMETRY ---")
print(best_result)






