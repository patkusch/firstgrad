"""Loss, the nudge step, and the loop that repeats them."""
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


def step(model, lr):
    """Move every number a little in the direction that lowers the loss."""
    for p in model.parameters():
        p.data -= lr * p.grad


def fit(model, xs, ys, loss_fn, lr=0.1, epochs=200, log_every=0):
    history = []
    for epoch in range(epochs):
        outputs = [model(x) for x in xs]
        loss = loss_fn(outputs, ys)
        model.zero_grad()
        loss.backward()
        step(model, lr)
        history.append(loss.data)
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:4d}  loss {loss.data:.4f}")
    return history


def accuracy(model, xs, ys):
    right = sum((model(x).data > 0) == (y == 1) for x, y in zip(xs, ys))
    return right / len(xs)
