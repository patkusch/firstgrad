"""Neurons, layers and a small multi-layer network built from Values."""
import random

from .value import Value


class Neuron:
    def __init__(self, n_in, activation, rng):
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        total = self.b
        for wi, xi in zip(self.w, x):
            total = total + wi * xi
        return getattr(total, self.activation)() if self.activation else total

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, activation, rng):
        self.neurons = [Neuron(n_in, activation, rng) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """sizes=[2, 8, 1] means 2 inputs, one hidden layer of 8, 1 output."""

    def __init__(self, sizes, activation="tanh", seed=0):
        rng = random.Random(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            self.layers.append(
                Layer(sizes[i], sizes[i + 1], None if last else activation, rng)
            )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
