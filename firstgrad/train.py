"""Loss, the nudge step, and the loop that repeats them."""
import random

from .value import Value


def squared_error(outputs, targets):
    """Average squared gap between answers and the right answers."""
    total = Value(0.0)
    for o, t in zip(outputs, targets):
        total = total + (o - t) ** 2
    return total * (1 / len(targets))


def log_loss(logits, labels):
    """Penalty for confident wrong answers. labels are 0 or 1."""
    total = Value(0.0)
    for z, y in zip(logits, labels):
        p = z.sigmoid()
        total = total - (y * (p + 1e-9).log() + (1 - y) * (1 - p + 1e-9).log())
    return total * (1 / len(labels))


def softmax(scores):
    """Turn ten raw scores into ten chances that add up to 1."""
    top = max(s.data for s in scores)
    exps = [(s - top).exp() for s in scores]
    total = exps[0]
    for e in exps[1:]:
        total = total + e
    return [e / total for e in exps]


def cross_entropy(score_lists, labels):
    """Penalty for giving the right answer a low chance. labels are 0..N-1."""
    total = Value(0.0)
    for scores, y in zip(score_lists, labels):
        total = total - softmax(scores)[y].log()
    return total * (1 / len(labels))


def step(model, lr):
    """Move every number a little in the direction that lowers the loss."""
    for p in model.parameters():
        p.data -= lr * p.grad


def fit(model, xs, ys, loss_fn, lr=0.1, epochs=200, log_every=0, optimizer=None,
        batch_size=None, seed=0):
    """Plain descent at rate lr unless an optimizer from optim.py is given.

    With batch_size, each nudge looks at a random handful of examples instead
    of all of them: noisier, but many more nudges per pass over the data.
    """
    history = []
    rng = random.Random(seed)
    order = list(range(len(xs)))
    for epoch in range(epochs):
        if batch_size:
            rng.shuffle(order)
            pick = order[:batch_size]
        else:
            pick = order
        outputs = [model(xs[i]) for i in pick]
        loss = loss_fn(outputs, [ys[i] for i in pick])
        model.zero_grad()
        loss.backward()
        if optimizer:
            optimizer.step()
        else:
            step(model, lr)
        history.append(loss.data)
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:4d}  loss {loss.data:.4f}")
    return history


def accuracy(model, xs, ys):
    right = sum((model(x).data > 0) == (y == 1) for x, y in zip(xs, ys))
    return right / len(xs)


def predict(model, x):
    """Index of the highest score."""
    scores = model(x)
    return max(range(len(scores)), key=lambda i: scores[i].data)


def class_accuracy(model, xs, ys):
    return sum(predict(model, x) == y for x, y in zip(xs, ys)) / len(xs)
