"""Small made-up datasets, so nothing needs downloading."""
import math
import random

XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [0, 1, 1, 0]


def two_spirals(n_per_arm=40, noise=0.05, seed=0, turns=1.5):
    """Two interleaved spirals. No straight line can split them."""
    rng = random.Random(seed)
    xs, ys = [], []
    for label in (0, 1):
        for i in range(n_per_arm):
            r = (i + 1) / n_per_arm
            angle = turns * math.pi * r + label * math.pi
            xs.append([
                r * math.cos(angle) + rng.gauss(0, noise),
                r * math.sin(angle) + rng.gauss(0, noise),
            ])
            ys.append(label)
    order = list(range(len(xs)))
    rng.shuffle(order)
    return [xs[i] for i in order], [ys[i] for i in order]
