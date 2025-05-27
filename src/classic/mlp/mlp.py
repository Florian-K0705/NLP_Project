import numpy as np
import classic.mlp.helpfunctions as help


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        # Xavier/He-Initialisierung
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = help.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = help.softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        
        # Output Layer Gradients
        dZ2 = y_pred.copy()
        dZ2[range(m), y_true] -= 1
        dZ2 /= m
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden Layer Gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU Ableitung
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = help.cross_entropy_loss(y, y_pred)
            self.backward(X, y, y_pred)
            preds = np.argmax(y_pred, axis=1)
            acc = help.accuracy(y, preds)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
