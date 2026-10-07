"""The three-turn spirals that plain descent could not solve.

Trains the same network three ways, then scores each on fresh spirals it has
never seen, so memorising can't pass for learning.
"""
from firstgrad.data import two_spirals
from firstgrad.nn import MLP
from firstgrad.optim import SGD, Adam, Momentum
from firstgrad.train import accuracy, fit, log_loss

train_x, train_y = two_spirals(n_per_arm=30, turns=3.0, seed=0)
test_x, test_y = two_spirals(n_per_arm=30, turns=3.0, seed=1)

for name, make in [("plain", lambda p: SGD(p, 0.3)),
                   ("momentum", lambda p: Momentum(p, 0.1)),
                   ("adam", lambda p: Adam(p, 0.1))]:
    model = MLP([2, 12, 12, 1], seed=3)
    fit(model, train_x, train_y, log_loss, epochs=400, optimizer=make(model.parameters()))
    print(f"{name:9s} seen {accuracy(model, train_x, train_y):.0%}   unseen {accuracy(model, test_x, test_y):.0%}")
