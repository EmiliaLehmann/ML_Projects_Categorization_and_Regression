import numpy as np


class DeepSpotifyNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, lr, activation, dropout_rate=0.2):
        self.lr = lr
        self.activation_type = activation
        self.dropout_rate = dropout_rate
        self.l2_lambda = 0.0001
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

        factor = 2.0 if activation in ['relu', 'leaky_relu'] else 1.0

        # Inicjalizacja wag dla 3 warstw ukrytych + wyjściowej
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(factor / input_size)
        self.b1 = np.zeros((1, hidden_size1))

        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(factor / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))

        self.W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(factor / hidden_size2)
        self.b3 = np.zeros((1, hidden_size3))

        self.W4 = np.random.randn(hidden_size3, output_size) * np.sqrt(factor / hidden_size3)
        self.b4 = np.zeros((1, output_size))

        # Momenty Adam dla wszystkich wag i biasów
        self.params = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']
        for p in self.params:
            setattr(self, f'm{p}', np.zeros_like(getattr(self, p)))
            setattr(self, f'v{p}', np.zeros_like(getattr(self, p)))

        self.loss_history = []

    def _activate(self, x):
        if self.activation_type == 'relu': return np.maximum(0, x)
        if self.activation_type == 'leaky_relu': return np.where(x > 0, x, x * 0.01)
        if self.activation_type == 'sigmoid': return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return np.tanh(x)

    def _derivative(self, a, z):
        if self.activation_type == 'relu': return (z > 0).astype(float)
        if self.activation_type == 'leaky_relu':
            dx = np.ones_like(z)
            dx[z <= 0] = 0.01
            return dx
        if self.activation_type == 'sigmoid': return a * (1 - a)
        return 1 - np.square(a)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, training=True):
        # Warstwa 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._activate(self.z1)
        if training:
            self.mask1 = (np.random.rand(*self.a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a1 *= self.mask1

        # Warstwa 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._activate(self.z2)
        if training:
            self.mask2 = (np.random.rand(*self.a2.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a2 *= self.mask2

        # Warstwa 3 (NOWA)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self._activate(self.z3)
        if training:
            self.mask3 = (np.random.rand(*self.a3.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a3 *= self.mask3

        # Warstwa wyjściowa
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.probs = self._softmax(self.z4)
        return self.probs

    def backward(self, X_batch, y_batch, probs, batch_size):
        # Gradient wyjściowy (Warstwa 4)
        dz4 = probs - y_batch
        dW4 = np.dot(self.a3.T, dz4) / batch_size + self.l2_lambda * self.W4
        db4 = np.sum(dz4, axis=0, keepdims=True) / batch_size

        # Gradient Warstwy 3
        dz3 = np.dot(dz4, self.W4.T) * self._derivative(self.a3, self.z3)
        if hasattr(self, 'mask3'): dz3 *= self.mask3
        dW3 = np.dot(self.a2.T, dz3) / batch_size + self.l2_lambda * self.W3
        db3 = np.sum(dz3, axis=0, keepdims=True) / batch_size

        # Gradient Warstwy 2
        dz2 = np.dot(dz3, self.W3.T) * self._derivative(self.a2, self.z2)
        if hasattr(self, 'mask2'): dz2 *= self.mask2
        dW2 = np.dot(self.a1.T, dz2) / batch_size + self.l2_lambda * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        # Gradient Warstwy 1
        dz1 = np.dot(dz2, self.W2.T) * self._derivative(self.a1, self.z1)
        if hasattr(self, 'mask1'): dz1 *= self.mask1
        dW1 = np.dot(X_batch.T, dz1) / batch_size + self.l2_lambda * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        # Adam Update
        self.t += 1
        def update(p, dp, mp, vp):
            mp[:] = self.beta1 * mp + (1 - self.beta1) * dp
            vp[:] = self.beta2 * vp + (1 - self.beta2) * (dp ** 2)
            m_hat = mp / (1 - self.beta1 ** self.t)
            v_hat = vp / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        update(self.W4, dW4, self.mW4, self.vW4); update(self.b4, db4, self.mb4, self.vb4)
        update(self.W3, dW3, self.mW3, self.vW3); update(self.b3, db3, self.mb3, self.vb3)
        update(self.W2, dW2, self.mW2, self.vW2); update(self.b2, db2, self.mb2, self.vb2)
        update(self.W1, dW1, self.mW1, self.vW1); update(self.b1, db1, self.mb1, self.vb1)

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        ce = -np.sum(y_true * np.log(y_pred)) / m
        l2 = self.l2_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2) + np.sum(self.W4**2))
        return ce + l2