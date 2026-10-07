"""XOR: output 1 when exactly one input is 1. A single neuron cannot do it."""
from firstgrad.data import XOR_X, XOR_Y
from firstgrad.nn import MLP
from firstgrad.train import accuracy, fit, log_loss

model = MLP([2, 4, 1], seed=1)
fit(model, XOR_X, XOR_Y, log_loss, lr=0.5, epochs=300, log_every=50)
for x, y in zip(XOR_X, XOR_Y):
    print(x, "->", round(model(x).sigmoid().data, 3), "(want", y, ")")
print("accuracy:", accuracy(model, XOR_X, XOR_Y))
