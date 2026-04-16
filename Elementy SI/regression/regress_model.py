import numpy as np

#Chat we are so cooked
class TemperaturePredictionNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, num_layers=1, activation='tanh'):
        self.hidden_size = hidden_size
        self.lr = lr
        self.num_layers = num_layers
        self.activation_type = activation

        # Inicjalizacja wag dla wielu warstw
        self.params = []  # Lista słowników z wagami dla każdej warstwy

        current_input_dim = input_size
        for i in range(num_layers):
            layer = {
                'Wxh': np.random.randn(hidden_size, current_input_dim) * 0.1,
                'Whh': np.random.randn(hidden_size, hidden_size) * 0.1,
                'bh': np.zeros((hidden_size, 1))
            }
            self.params.append(layer)
            current_input_dim = hidden_size  # Wyjście warstwy i jest wejściem dla i+1

        # Warstwa wyjściowa (Dense)
        self.Why = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

    def _activate(self, x):
        if self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'relu':
            return np.maximum(0, x)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'elu':
            # ELU: x jeśli x > 0, inaczej 1.0 * (exp(x) - 1)
            return np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
        return x
    #zmieniamy funkcje aktywacji w sensie testujemy rip
    def _activate_derivative(self, activated_x):
        if self.activation_type == 'tanh':
            return 1 - activated_x ** 2
        elif self.activation_type == 'relu':
            return (activated_x > 0).astype(float)
        elif self.activation_type == 'sigmoid':
            return activated_x * (1 - activated_x)
        elif self.activation_type == 'elu':
            # Pochodna ELU: 1 jeśli x > 0, inaczej f(x) + alpha
            # Ponieważ przekazujemy już 'activated_x' (wynik funkcji f(x)):
            return np.where(activated_x > 0, 1.0, activated_x + 1.0)
        return np.ones_like(activated_x)

    def forward(self, x_sequence):
        # h_states[warstwa][krok_czasowy]
        h_states = [{-1: np.zeros((self.hidden_size, 1))} for _ in range(self.num_layers)]

        current_input = [xt.reshape(-1, 1) for xt in x_sequence]

        for l in range(self.num_layers):
            next_input = []
            for t in range(len(current_input)):
                z = np.dot(self.params[l]['Wxh'], current_input[t]) + \
                    np.dot(self.params[l]['Whh'], h_states[l][t - 1]) + \
                    self.params[l]['bh']
                h_states[l][t] = self._activate(z)
                next_input.append(h_states[l][t])
            current_input = next_input  # Przekaż do następnej warstwy

        y_pred = np.dot(self.Why, h_states[-1][len(x_sequence) - 1]) + self.by
        return y_pred, h_states

    def train(self, x_sequence, y_true):
        y_pred, h_states = self.forward(x_sequence)
        dy = y_pred - y_true.reshape(-1, 1)

        # Gradienty dla warstwy wyjściowej
        dWhy = np.dot(dy, h_states[-1][len(x_sequence) - 1].T)
        dby = dy

        # Propagacja błędu wstecz przez warstwy (Backpropagation Through Time & Layers)
        dh_next_layer = np.dot(self.Why.T, dy)

        for l in reversed(range(self.num_layers)):
            dWxh, dWhh, dbh = 0, 0, 0
            dh_next_step = np.zeros((self.hidden_size, 1))

            # Pobierz wejście, które wchodziło do tej konkretnej warstwy
            if l == 0:
                layer_input = [xt.reshape(-1, 1) for xt in x_sequence]
            else:
                layer_input = [h_states[l - 1][t] for t in range(len(x_sequence))]

            for t in reversed(range(len(x_sequence))):
                # Gradient płynie od warstwy wyżej ORAZ od następnego kroku czasowego
                dtanh = self._activate_derivative(h_states[l][t]) * (dh_next_layer + dh_next_step)

                dbh += dtanh
                dWxh += np.dot(dtanh, layer_input[t].T)
                dWhh += np.dot(dtanh, h_states[l][t - 1].T)

                dh_next_step = np.dot(self.params[l]['Whh'].T, dtanh)
                # Błąd dla warstwy poniżej w tym samym kroku czasowym
                # w Stacked RNN błąd idzie też "w dół")

            # Aktualizacja wag warstwy l
            self.params[l]['Wxh'] -= self.lr * np.clip(dWxh, -5, 5)
            self.params[l]['Whh'] -= self.lr * np.clip(dWhh, -5, 5)
            self.params[l]['bh'] -= self.lr * np.clip(dbh, -5, 5)

        # Aktualizacja wag wyjściowych
        self.Why -= self.lr * np.clip(dWhy, -5, 5)
        self.by -= self.lr * np.clip(dby, -5, 5)