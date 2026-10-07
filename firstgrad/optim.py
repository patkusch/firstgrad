"""Ways of deciding how far to nudge each number.

Plain descent steps straight down the slope. Momentum keeps some of its
previous direction, so it rolls through flat patches. Adam also gives each
number its own step size, small where the slope is wild and large where it is
gentle.
"""
import math


class SGD:
    def __init__(self, params, lr=0.1):
        self.params, self.lr = list(params), lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad


class Momentum:
    def __init__(self, params, lr=0.05, beta=0.9):
        self.params, self.lr, self.beta = list(params), lr, beta
        self.v = [0.0] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            self.v[i] = self.beta * self.v[i] + p.grad
            p.data -= self.lr * self.v[i]


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = list(params), lr, b1, b2, eps
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
