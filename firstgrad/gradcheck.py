"""Proof that the gradients are right.

Nudges each input by a tiny amount in both directions, measures how the output
moves, and compares that slow-but-obviously-correct answer with what backward()
computed.
"""


def numeric_grad(f, inputs, index, eps=1e-6):
    """f takes a list of plain floats and returns a float."""
    up = list(inputs)
    down = list(inputs)
    up[index] += eps
    down[index] -= eps
    return (f(up) - f(down)) / (2 * eps)


def max_gradient_error(f, inputs):
    """Largest gap between backward() and the nudge-and-measure answer.

    f takes Values and returns a Value.
    """
    from .value import Value

    vals = [Value(x) for x in inputs]
    f(vals).backward()
    worst = 0.0
    for i, v in enumerate(vals):
        slow = numeric_grad(lambda xs: f([Value(x) for x in xs]).data, inputs, i)
        worst = max(worst, abs(slow - v.grad))
    return worst
