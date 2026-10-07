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


_DIGITS = """
.###. #...# #..## #.#.# ##..# #...# .###.
..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.
.###. #...# ....# ...#. ..#.. .#... #####
##### ...#. ..#.. ...#. ....# #...# .###.
...#. ..##. .#.#. #..#. ##### ...#. ...#.
##### #.... ####. ....# ....# #...# .###.
..##. .#... #.... ####. #...# #...# .###.
##### ....# ...#. ..#.. .#... .#... .#...
.###. #...# #...# .###. #...# #...# .###.
.###. #...# #...# .#### ....# ...#. .##..
"""

# ten hand-drawn digits, each 7 rows of 5 pixels, flattened to 35 numbers
DIGIT_BITMAPS = [
    [1.0 if ch == "#" else 0.0 for row in line.split() for ch in row]
    for line in _DIGITS.strip().splitlines()
]


def noisy_digits(copies=8, flip=0.08, seed=0):
    """Each digit `copies` times, every pixel flipped with chance `flip`."""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(copies):
        for digit, bitmap in enumerate(DIGIT_BITMAPS):
            xs.append([1 - p if rng.random() < flip else p for p in bitmap])
            ys.append(digit)
    order = list(range(len(xs)))
    rng.shuffle(order)
    return [xs[i] for i in order], [ys[i] for i in order]
