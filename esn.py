import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os

ACTIVATIONS = {
    "tanh": np.tanh,
    "sigmoid": lambda x: 1/(1 + np.exp(-x)),
    "none": None
}

EPSILON = 1e-7

class ESN:
    def __init__(self, size, density=0.1, spectral_radius=0.95, bias=False, norm_rate=None, activation="tanh"):
        self.size = size
        self.density = density
        self.spectral_radius = spectral_radius
        self.bias = bias
        self.activation_name = activation
        self.activation = ACTIVATIONS[activation.lower()]
        self.norm_rate = norm_rate

        self.state = np.zeros(size)
        self.mean = np.zeros(size)
        self.deviation = np.ones(size)

        #create weights
        w_size = size + 1 if bias else size
        self.weights = np.random.normal(0, 1, size=[w_size, w_size])
        #sparsify
        self.weights[np.random.random(size=self.weights.shape) < (1 - density)] = 0
        #match desired spectral radius
        r = np.max(np.abs(np.linalg.eigvals(self.weights)))
        self.weights = self.weights * (spectral_radius / r)
        self.weights = csr_matrix(self.weights)

    def step(self):
        x = self.state

        if self.norm_rate is not None:
            delta = x - self.mean
            x = delta / (self.deviation + EPSILON)

            self.mean += delta * self.norm_rate
            self.deviation += (np.abs(delta) - self.deviation) * self.norm_rate

        if self.bias:
            x = np.append(x, [1])

        x = self.weights @ x
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
                    "norm_rate": self.norm_rate
                },
                "weights": self.weights,
                "state": self.state,
                "mean": self.mean,
                "deviation": self.deviation
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
        esn.mean = data["mean"] if "mean" in data else np.zeros_like(esn.state)
        esn.deviation = data["deviation"] if "deviation" in data else np.ones_like(esn.state)
        return esn
