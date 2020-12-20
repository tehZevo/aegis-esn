import numpy as np
import pickle
import os

ACTIVATIONS = {
    "tanh": np.tanh,
    "sigmoid": lambda x: 1/(1 + np.exp(-x)),
    "none": None
}

class ESN:
    def __init__(self, size, density=0.1, spectral_radius=0.95, bias=True, activation="tanh"):
        self.size = size
        self.density = density
        self.spectral_radius = spectral_radius
        self.bias = bias
        self.activation_name = activation
        self.activation = ACTIVATIONS[activation.lower()]

        self.state = np.zeros(size)

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

    def save(self, filepath):
        # os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "params": {
                    "size": self.size,
                    "density": self.density,
                    "spectral_radius": self.spectral_radius,
                    "bias": self.bias,
                    "activation": self.activation_name,
                },
                "weights": self.weights,
                "state": self.state
            }, f)

    def load(filepath):
        #load data
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        #create esn with same params
        esn = ESN(**data["params"])
        #set weights and current state
        esn.weights = data["weights"]
        esn.state = data["state"]
        #return esn
        return esn
