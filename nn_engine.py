import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        # He/Xavier-style initialization scaled for stability
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((output_dim,))
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        self.x = x
        if x.ndim == 1:
            return np.dot(x, self.W) + self.b
        return np.dot(x, self.W) + self.b

    def backward(self, upstream_grad):
        if self.x.ndim == 1:
            x_reshaped = self.x.reshape(1, -1)
        else:
            x_reshaped = self.x
            
        if upstream_grad.ndim == 1:
            grad_reshaped = upstream_grad.reshape(1, -1)
        else:
            grad_reshaped = upstream_grad

        self.dW = np.dot(x_reshaped.T, grad_reshaped)
        self.db = np.sum(grad_reshaped, axis=0)
        dx = np.dot(grad_reshaped, self.W.T)
        
        return dx.squeeze() if self.x.ndim == 1 else dx


class Sigmoid:
    def forward(self, z):
        # Numerically stable sigmoid or standard
        self.out = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return self.out

    def backward(self, upstream_grad):
        return upstream_grad * self.out * (1.0 - self.out)


class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        # Derivative of MSE: 2 * (y_pred - y_true) / n
        n = self.y_pred.size
        return (2.0 / n) * (self.y_pred - self.y_true)


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class SGD:
    def __init__(self, layers, lr=0.5):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.dW is not None:
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db