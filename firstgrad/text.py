"""Letters as numbers, and the simplest possible name-writer: count which
letter tends to follow which."""
import math
import random

END = "."  # marks both the start and the end of a name
LETTERS = [END] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
INDEX = {ch: i for i, ch in enumerate(LETTERS)}


def pairs(names):
    """(letter, next letter) for every step of every name, as numbers."""
    out = []
    for name in names:
        chars = END + name + END
        out.extend((INDEX[a], INDEX[b]) for a, b in zip(chars, chars[1:]))
    return out


def count_table(names, smoothing=1.0):
    """chances[a][b] = how likely letter b follows letter a, from counting."""
    n = len(LETTERS)
    counts = [[smoothing] * n for _ in range(n)]
    for a, b in pairs(names):
        counts[a][b] += 1
    return [[c / sum(row) for c in row] for row in counts]


def average_surprise(chances, names):
    """Average penalty (log loss) for the real next letters. Lower is better.

    Always picking among 27 letters at random scores ln(27), about 3.30.
    """
    ps = pairs(names)
    return -sum(math.log(chances[a][b]) for a, b in ps) / len(ps)


def write_name(next_letter_chances, rng, limit=15):
    """Write one name by repeatedly drawing the next letter."""
    current, out = INDEX[END], ""
    while len(out) < limit:
        weights = next_letter_chances(current)
        current = rng.choices(range(len(LETTERS)), weights=weights)[0]
        if current == INDEX[END]:
            break
        out += LETTERS[current]
    return out


def one_hot(i):
    v = [0.0] * len(LETTERS)
    v[i] = 1.0
    return v


def network_chances(model):
    """Wrap a trained network so it answers like count_table: letter -> chances."""
    from .train import softmax

    return lambda a: [p.data for p in softmax(model(one_hot(a)))]


def network_surprise(model, names):
    """Same score as average_surprise, for a network. Lower is better."""
    ask = network_chances(model)
    ps = pairs(names)
    return -sum(math.log(ask(a)[b]) for a, b in ps) / len(ps)
