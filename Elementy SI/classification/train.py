import numpy as np


def train_model(model, X_train, y_train_oh, epochs=50, batch_size=128):
    m = X_train.shape[0]
    for epoch in range(epochs):
        perm = np.random.permutation(m)
        X_shuffled = X_train[perm]
        y_shuffled = y_train_oh[perm]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            probs = model.forward(X_batch)
            current_batch_size = X_batch.shape[0]
            model.backward(X_batch, y_batch, probs, current_batch_size)

        # Obliczanie straty na koniec epoki
        full_probs = model.forward(X_train)
        loss = -np.mean(np.sum(y_train_oh * np.log(full_probs + 1e-10), axis=1))
        model.loss_history.append(loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return model

