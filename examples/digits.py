"""Read hand-drawn digits, then see how much static it can take.

Trains on digits with 8% of pixels flipped, then scores on fresh digits at
rising noise levels and shows which digits get mixed up.
"""
from collections import Counter

from firstgrad.data import noisy_digits
from firstgrad.nn import MLP
from firstgrad.optim import Adam
from firstgrad.train import class_accuracy, cross_entropy, fit, predict

xs, ys = noisy_digits(copies=8, flip=0.08, seed=0)
model = MLP([35, 16, 10], seed=2)
fit(model, xs, ys, cross_entropy, epochs=100, batch_size=20, log_every=25,
    optimizer=Adam(model.parameters(), lr=0.05))

print("\nstatic    right on fresh digits")
for flip in (0.0, 0.05, 0.10, 0.15, 0.25, 0.35):
    tx, ty = noisy_digits(copies=20, flip=flip, seed=100)
    print(f"{flip:5.0%}     {class_accuracy(model, tx, ty):.0%}")

tx, ty = noisy_digits(copies=40, flip=0.15, seed=200)
mixups = Counter((y, predict(model, x)) for x, y in zip(tx, ty) if predict(model, x) != y)
print("\nmost common mix-ups at 15% static (true -> guessed):")
for (true, guess), n in mixups.most_common(4):
    print(f"  {true} -> {guess}   {n} times")
