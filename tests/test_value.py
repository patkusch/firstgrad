import math
import random
import unittest

from firstgrad.gradcheck import max_gradient_error
from firstgrad.value import Value

TOL = 1e-6


class TestValue(unittest.TestCase):
    def test_known_answer(self):
        # d/dx of x*y + x is y + 1; d/dy is x
        x, y = Value(3.0), Value(4.0)
        (x * y + x).backward()
        self.assertEqual(x.grad, 5.0)
        self.assertEqual(y.grad, 3.0)

    def test_value_used_twice_adds_up(self):
        x = Value(2.0)
        (x * x).backward()
        self.assertEqual(x.grad, 4.0)

    def test_every_operation_matches_brute_force(self):
        cases = {
            "add": lambda v: v[0] + v[1],
            "sub": lambda v: v[0] - v[1],
            "mul": lambda v: v[0] * v[1],
            "div": lambda v: v[0] / v[1],
            "pow": lambda v: v[0] ** 3,
            "neg": lambda v: -v[0],
            "exp": lambda v: v[0].exp(),
            "log": lambda v: v[1].log(),
            "tanh": lambda v: v[0].tanh(),
            "sigmoid": lambda v: v[0].sigmoid(),
            "relu": lambda v: v[0].relu(),
            "rsub": lambda v: 5 - v[0],
            "rdiv": lambda v: 5 / v[1],
            "chain": lambda v: (v[0] * v[1] + 1).tanh() * v[1].exp(),
        }
        for name, f in cases.items():
            with self.subTest(op=name):
                self.assertLess(max_gradient_error(f, [0.7, 1.3]), TOL)

    def test_relu_is_flat_below_zero(self):
        x = Value(-2.0)
        x.relu().backward()
        self.assertEqual(x.grad, 0.0)

    def test_random_expressions(self):
        rng = random.Random(1)
        for trial in range(25):
            a, b, c = (rng.uniform(0.3, 2.0) for _ in range(3))
            f = lambda v: ((v[0] * v[1] - v[2]).tanh() + v[0] / v[2]) ** 2
            self.assertLess(max_gradient_error(f, [a, b, c]), 1e-5, trial)

    def test_backward_twice_does_not_double_count(self):
        x = Value(3.0)
        y = x * x
        y.backward()
        y.backward()
        self.assertEqual(x.grad, 6.0)

    def test_values_match_math(self):
        self.assertAlmostEqual(Value(0.5).tanh().data, math.tanh(0.5))
        self.assertAlmostEqual(Value(0.5).sigmoid().data, 1 / (1 + math.exp(-0.5)))


if __name__ == "__main__":
    unittest.main()
