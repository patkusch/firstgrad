"""Write made-up names two ways: by counting, and with a trained network.

Both are scored on names they never saw. Lower surprise is better; picking
among 27 symbols blindly scores 3.30.
"""
import math
import random

from firstgrad.data import NAMES
from firstgrad.nn import MLP
from firstgrad.optim import Adam
from firstgrad.text import (LETTERS, average_surprise, count_table, network_chances,
                            network_surprise, one_hot, pairs, write_name)
from firstgrad.train import cross_entropy, fit

train, held_out = NAMES[::2], NAMES[1::2]
table = count_table(train)

ps = pairs(train)
model = MLP([len(LETTERS), 20, len(LETTERS)], seed=4)
fit(model, [one_hot(a) for a, _ in ps], [b for _, b in ps], cross_entropy,
    epochs=100, batch_size=64, optimizer=Adam(model.parameters(), lr=0.05))

print(f"blind guessing   {math.log(len(LETTERS)):.2f}")
print(f"counting         {average_surprise(table, held_out):.2f}")
print(f"network          {network_surprise(model, held_out):.2f}\n")

for label, ask in [("counting", lambda a: table[a]), ("network", network_chances(model))]:
    rng = random.Random(7)
    print(label, ":", ", ".join(write_name(ask, rng) or "-" for _ in range(10)))
