import numpy as np

class DeepSpotifyNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lr, activation,  dropout_rate=0.2):
        self.lr = lr
        self.activation_type = activation
        self.dropout_rate = dropout_rate

        factor = 2.0 if activation in ['relu', 'leaky_relu'] else 1.0

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0




        # Inicjalizacja wag
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(factor / input_size)
        self.b1 = np.zeros((1, hidden_size1))

        self.l2_lambda = 0.0001

        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(factor / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        # self.W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2. / hidden_size2)
        # self.b3 = np.zeros((1, hidden_size3))

        self.W4 = np.random.randn(hidden_size2, output_size) * np.sqrt(factor / hidden_size2)
        self.b4 = np.zeros((1, output_size))

        # momenty
        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)

        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)

        self.mW4 = np.zeros_like(self.W4)
        self.vW4 = np.zeros_like(self.W4)

        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)

        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)

        self.mb4 = np.zeros_like(self.b4)
        self.vb4 = np.zeros_like(self.b4)

        self.loss_history = []

    def _activate(self, x):
        if self.activation_type == 'relu':
            return np.maximum(0, x)
        elif self.activation_type == 'leaky_relu':
            return np.where(x > 0, x, x * 0.01)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation_type == 'tanh':
            return np.tanh(x)

    def _derivative(self, a, z):
        if self.activation_type == 'relu':
            return (z > 0).astype(float)
        elif self.activation_type == 'leaky_relu':
            dx = np.ones_like(z)
            dx[z <= 0] = 0.01
            return dx
        elif self.activation_type == 'sigmoid':
            return a * (1 - a)
        elif self.activation_type == 'tanh':
            return 1 - np.square(a)


    def compute_loss(self, y_true, y_pred):
        # Cross-Entropy Loss
        m = y_true.shape[0]

        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)

        cross_entropy = -np.sum(y_true * np.log(y_pred)) / m

        l2 = self.l2_lambda * (
                np.sum(self.W1 ** 2) +
                np.sum(self.W2 ** 2) +
                np.sum(self.W4 ** 2)
        )
        return cross_entropy + l2

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, training=True):

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._activate(self.z1)

        if training:
            self.dropout_mask1 = (np.random.rand(*self.a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a1 *= self.dropout_mask1

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._activate(self.z2)
        # self.z3 = np.dot(self.a2, self.W3) + self.b3
        # self.a3 = self._relu(self.z3)
        if training:
            self.dropout_mask2 = (np.random.rand(*self.a2.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a2 *= self.dropout_mask2

        self.z4 = np.dot(self.a2, self.W4) + self.b4
        self.probs = self._softmax(self.z4)
        return self.probs

    def backward(self, X_batch, y_batch, probs, batch_size):

        dz4 = probs - y_batch
        dW4 = np.dot(self.a2.T, dz4) / batch_size
        dW4 += self.l2_lambda * self.W4
        db4 = np.sum(dz4, axis=0, keepdims=True) / batch_size

        # dz3 = np.dot(dz4, self.W4.T) * (self.z3 > 0)
        # dW3 = np.dot(self.a2.T, dz3) / batch_size
        # db3 = np.sum(dz3, axis=0, keepdims=True) / batch_size

        dz2 = np.dot(dz4, self.W4.T) * self._derivative(self.a2, self.z2)
        if hasattr(self, 'dropout_mask2'):
            dz2 *= self.dropout_mask2

        dW2 = np.dot(self.a1.T, dz2) / batch_size
        dW2 += self.l2_lambda * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size


        dz1 = np.dot(dz2, self.W2.T) * self._derivative(self.a1, self.z1)
        dz1 *= self.dropout_mask1

        dW1 = np.dot(X_batch.T, dz1) / batch_size
        dW1 += self.l2_lambda * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size


        self.t += 1

        def adam_update(param, dparam, m, v):
            m[:] = self.beta1 * m + (1 - self.beta1) * dparam
            v[:] = self.beta2 * v + (1 - self.beta2) * (dparam ** 2)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Wagi
        adam_update(self.W1, dW1, self.mW1, self.vW1)
        adam_update(self.W2, dW2, self.mW2, self.vW2)
        adam_update(self.W4, dW4, self.mW4, self.vW4)

        # Biasy
        adam_update(self.b1, db1, self.mb1, self.vb1)
        adam_update(self.b2, db2, self.mb2, self.vb2)
        adam_update(self.b4, db4, self.mb4, self.vb4)


        # # Update wag
        # self.W1 -= self.lr * dW1
        # self.b1 -= self.lr * db1
        #
        # self.W2 -= self.lr * dW2
        # self.b2 -= self.lr * db2
        # # self.W3 -= self.lr * dW3
        # # self.b3 -= self.lr * db3
        #
        # self.W4 -= self.lr * dW4
        # self.b4 -= self.lr * db4


