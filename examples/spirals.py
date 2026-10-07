"""Two spirals, and a picture of where the network draws the line."""
from firstgrad.data import two_spirals
from firstgrad.nn import MLP
from firstgrad.train import accuracy, fit, log_loss

xs, ys = two_spirals(n_per_arm=30)
model = MLP([2, 12, 12, 1], seed=3)
fit(model, xs, ys, log_loss, lr=0.3, epochs=300, log_every=50)
print("accuracy:", accuracy(model, xs, ys))

for row in range(21):
    y = 1.1 - row * 0.11
    print("".join("#" if model([-1.1 + c * 0.055, y]).data > 0 else "." for c in range(41)))
