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


#to jest inny typ doboru danych (walk forward validation) - nie random wybierane jak w naszym modelu z kategoryzacja tylko duzo na raz po sobie
#no bo ma przewidywac w czasie po sobie a nie random ahh dane z 2021 a potem 1939

#no tu w sumie parametry do modelu dobieramy
window_size = 12
hidden_size = 64
initial_lr = 0.001
train_size = 1200

model = TemperaturePredictionNet(input_size=1, hidden_size=hidden_size, output_size=1, lr=initial_lr)

#to jest przygotowanie tabel do wyswietlenia wynikow na plocie
wf_predictions = []
wf_actuals = []

print("Starting Walk-Forward Validation :D")

# no start tego procesu jak print powyzej mowi
for i in range(train_size, len(scaled_data)):
    # wybieramy obecne okno - te 12 miesiecy jakies
    x_input = scaled_data[i - window_size:i].reshape(window_size, 1)
    y_true = scaled_data[i]

    #proba przewidzenia temp w nastepnym miesiacu po podanych 12stu z okna bez uczenia
    y_pred, _ = model.forward(x_input)
    wf_predictions.append(y_pred[0][0])
    wf_actuals.append(y_true)

    # teraz bierze dane z tych 12 miesiecy i na nich bazuje
    model.train(x_input, np.array([y_true]))

    # postep co kazde 120 miesiecy, my mamy wziete ich mega duzo btw bo train size to 1200
    if (i - train_size) % 120 == 0:
        error = np.abs(y_pred[0][0] - y_true)
        print(f"Month {i}: Current absolute error: {error:.4f}")

print("Validation Complete.")


def get_final_metrics(preds, actuals, d_min, d_max):
    #przerobienie z -1 do 1 na stopnie celicjusza czyli odwracamy poczatkowe skalowanie zeby moc odczytac po ludzku 15 stopni nie jakies 0.4516789120
    def denorm(x):
        return ((x + 1) / 2) * (d_max - d_min) + d_min

    p = denorm(np.array(preds))
    a = denorm(np.array(actuals))

    mae = np.mean(np.abs(p - a))
    print(f"\n--- Final Performance ---")
    #no i tu nam mowi o tym bledzie absolutnym czyli o ile sie jebnal w przewidzeniu z prawdziwa temperatura - usrednione
    print(f"Mean Absolute Error: {mae:.4f} °C")
    return p, a

#przygotowanie danych do wsadzenia do wykresu ktory ladnie pokazuje o ile nasz model sie myli
p_final, a_final = get_final_metrics(wf_predictions, wf_actuals, data_min, data_max)

import matplotlib.pyplot as plt

# wykres dla bledu na ostatnie 24 miesiace pzdr plt pd i np saving our asses here
plt.figure(figsize=(12, 6))
plt.plot(a_final[-24:], label='Actual Temp', marker='o', color='blue')
plt.plot(p_final[-24:], label='RNN Prediction', marker='x', linestyle='--', color='red')
plt.title('Kaggle Climate Data: Last 2 Years of Walk-Forward Predictions')
plt.ylabel('Degrees Celsius (°C)')
plt.xlabel('Months')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#bardzo lubie qsmp polska husaria btw wlasnie dyskutuja ile kosztuja bilety na malte
#o czym oni gadaja o linkedin HELP czy nexe i ewron tez studiowali informatyke XDDD