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


def windows(names, k):
    """(the k letters before, the letter that came next) for every step.

    Names are padded at the front with k start marks, so the first letter is
    predicted from "nothing yet" rather than skipped.
    """
    out = []
    for name in names:
        chars = END * k + name + END
        for i in range(k, len(chars)):
            out.append((tuple(INDEX[c] for c in chars[i - k:i]), INDEX[chars[i]]))
    return out


def one_hot_window(context):
    """Glue one one-hot letter per position into a single list of numbers."""
    v = []
    for i in context:
        v.extend(one_hot(i))
    return v


def count_table_k(names, k, smoothing=1.0):
    """Counting baseline that looks k letters back. Returns context -> chances."""
    n = len(LETTERS)
    counts = {}
    for ctx, nxt in windows(names, k):
        counts.setdefault(ctx, [smoothing] * n)[nxt] += 1
    blank = [1.0 / n] * n

    def ask(ctx):
        row = counts.get(tuple(ctx))
        return [c / sum(row) for c in row] if row else blank

    return ask


def surprise_k(ask, names, k):
    """Average log loss of ask(context) -> chances over every step of names."""
    ws = windows(names, k)
    return -sum(math.log(ask(c)[b]) for c, b in ws) / len(ws)


def write_name_k(ask, rng, k, limit=15):
    ctx, out = (INDEX[END],) * k, ""
    while len(out) < limit:
        nxt = rng.choices(range(len(LETTERS)), weights=ask(ctx))[0]
        if nxt == INDEX[END]:
            break
        out += LETTERS[nxt]
        ctx = ctx[1:] + (nxt,)
    return out
