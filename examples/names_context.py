"""Does remembering more letters help? Counting against a network, 1 to 3 back.

Both are scored on names they never saw. Lower surprise is better, and blind
guessing scores 3.30. "Seen" is the same score on the names they trained on:
a big gap between seen and unseen means it memorised instead of learned.
"""
import random

from firstgrad.data import NAMES
from firstgrad.nn import MLP
from firstgrad.optim import Adam
from firstgrad.text import (LETTERS, count_table_k, one_hot_window, surprise_k,
                            windows, write_name_k)
from firstgrad.train import cross_entropy, fit, softmax

train, held_out = NAMES[::2], NAMES[1::2]

print("letters back   counting (unseen)   network (seen)   network (unseen)")
samples = {}
for k in (1, 2, 3):
    ws = windows(train, k)
    model = MLP([len(LETTERS) * k, 30, len(LETTERS)], seed=4)
    fit(model, [one_hot_window(c) for c, _ in ws], [b for _, b in ws], cross_entropy,
        epochs=150, batch_size=64, optimizer=Adam(model.parameters(), lr=0.03))
    ask = lambda ctx, m=model: [p.data for p in softmax(m(one_hot_window(ctx)))]
    counted = surprise_k(count_table_k(train, k), held_out, k)
    print(f"{k:^12d}   {counted:^17.2f}   {surprise_k(ask, train, k):^14.2f}   {surprise_k(ask, held_out, k):^16.2f}")
    rng = random.Random(7)
    samples[k] = ", ".join(write_name_k(ask, rng, k) or "-" for _ in range(8))

print()
for k, names in samples.items():
    print(f"network, {k} back: {names}")
