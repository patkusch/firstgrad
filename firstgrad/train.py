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
        batch_size=None, seed=0, weight_decay=0.0):
    """Plain descent at rate lr unless an optimizer from optim.py is given.

    With batch_size, each nudge looks at a random handful of examples instead
    of all of them: noisier, but many more nudges per pass over the data.

    weight_decay shrinks every number a little towards zero after each nudge
    (0.01 means one per cent). Big numbers are how a network memorises odd
    cases, so a steady pull towards zero makes it keep only what many examples
    agree on.
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
        if weight_decay:
            for p in model.parameters():
                p.data *= 1 - weight_decay
        history.append(loss.data)
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:4d}  loss {loss.data:.4f}")
    return history


def fit_until_it_stops_helping(model, xs, ys, held_xs, held_ys, loss_fn, *, patience=20, check_every=5,
                               **fit_kwargs):
    """Train, but watch the loss on names (or anything) the network never trains on, and stop when that stops
    falling. The network is put back to how it was at its best moment.

    Memorising shows up as the training loss falling while the held-out loss rises. This catches the turn.
    Returns (training history, held-out history, the epoch the network was put back to).
    """
    history, held_history = [], []
    best, best_epoch, best_params, since = float("inf"), 0, None, 0
    epochs = fit_kwargs.pop("epochs", 200)
    for start in range(0, epochs, check_every):
        history += fit(model, xs, ys, loss_fn, epochs=min(check_every, epochs - start), seed=start, **fit_kwargs)
        held = loss_fn([model(x) for x in held_xs], held_ys).data
        held_history.append(held)
        if held < best:
            best, best_epoch, since = held, len(history), 0
            best_params = [p.data for p in model.parameters()]
        else:
            since += check_every
            if since >= patience:
                break
    for p, saved in zip(model.parameters(), best_params):
        p.data = saved
    return history, held_history, best_epoch


def accuracy(model, xs, ys):
    right = sum((model(x).data > 0) == (y == 1) for x, y in zip(xs, ys))
    return right / len(xs)


def predict(model, x):
    """Index of the highest score."""
    scores = model(x)
    return max(range(len(scores)), key=lambda i: scores[i].data)


def class_accuracy(model, xs, ys):
    return sum(predict(model, x) == y for x, y in zip(xs, ys)) / len(xs)
