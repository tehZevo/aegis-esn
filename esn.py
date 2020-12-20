import numpy as np

ACTIVATIONS = {
    "tanh": np.tanh,
    "sigmoid": lambda x: 1/(1 + np.exp(-x)),
    "none": None
}

class ESN:
    def __init__(self, size, density=0.1, spectral_radius=0.95, bias=True, activation="tanh"):
        self.size = size
        self.activation = ACTIVATIONS[activation.lower()]
        self.state = np.zeros(size)
        self.bias = bias

        #create weights
        w_size = size + 1 if bias else size
        self.weights = np.random.normal(0, 1, size=[w_size, w_size])
        #sparsify
        self.weights[np.random.random(size=self.weights.shape) < (1 - density)] = 0
        #match desired spectral radius
        r = np.max(np.abs(np.linalg.eigvals(self.weights)))
        self.weights = self.weights * (spectral_radius / r)

    def step(self):
        x = self.state

        if self.bias:
            x = np.append(x, [1])

        x = np.matmul(x, self.weights)
        if self.activation is not None:
            x = self.activation(x)

        if self.bias:
            x = x[:-1]

        self.state = x

    def reset(self):
        self.state = np.zeros_like(self.state)
