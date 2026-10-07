import unittest

from firstgrad.data import XOR_X, XOR_Y, two_spirals
from firstgrad.nn import MLP
from firstgrad.train import accuracy, fit, log_loss, squared_error


class TestNetworkGradients(unittest.TestCase):
    def test_every_weight_matches_brute_force(self):
        model = MLP([2, 3, 1], seed=5)
        x, target = [0.4, -0.8], 1.0
        loss = squared_error([model(x)], [target])
        model.zero_grad()
        loss.backward()

        eps = 1e-6
        for p in model.parameters():
            claimed = p.grad
            original = p.data
            p.data = original + eps
            up = squared_error([model(x)], [target]).data
            p.data = original - eps
            down = squared_error([model(x)], [target]).data
            p.data = original
            self.assertAlmostEqual(claimed, (up - down) / (2 * eps), places=5)


class TestLearning(unittest.TestCase):
    def test_loss_goes_down(self):
        model = MLP([2, 4, 1], seed=1)
        history = fit(model, XOR_X, XOR_Y, log_loss, lr=0.5, epochs=100)
        self.assertLess(history[-1], history[0] / 2)

    def test_learns_xor(self):
        model = MLP([2, 4, 1], seed=1)
        fit(model, XOR_X, XOR_Y, log_loss, lr=0.5, epochs=300)
        self.assertEqual(accuracy(model, XOR_X, XOR_Y), 1.0)

    def test_same_seed_gives_same_result(self):
        a = MLP([2, 4, 1], seed=9)
        b = MLP([2, 4, 1], seed=9)
        self.assertEqual(a([0.3, 0.1]).data, b([0.3, 0.1]).data)

    def test_spirals_beat_guessing_by_a_wide_margin(self):
        xs, ys = two_spirals(n_per_arm=30)
        model = MLP([2, 12, 12, 1], seed=3)
        fit(model, xs, ys, log_loss, lr=0.3, epochs=300)
        self.assertGreater(accuracy(model, xs, ys), 0.85)


if __name__ == "__main__":
    unittest.main()


class TestOptimizers(unittest.TestCase):
    def test_first_step_matches_hand_arithmetic(self):
        from firstgrad.optim import SGD, Adam, Momentum
        from firstgrad.value import Value

        for make, expected in [
            (lambda p: SGD(p, 0.1), 1.0 - 0.1 * 4.0),
            (lambda p: Momentum(p, 0.1, 0.9), 1.0 - 0.1 * 4.0),
            (lambda p: Adam(p, 0.1), 1.0 - 0.1),  # first Adam step is lr in size
        ]:
            w = Value(1.0)
            w.grad = 4.0
            make([w]).step()
            self.assertAlmostEqual(w.data, expected, places=6)

    def test_adam_solves_what_plain_descent_cannot(self):
        from firstgrad.optim import SGD, Adam

        xs, ys = two_spirals(n_per_arm=30, turns=3.0)
        scores = {}
        for name, make in [("sgd", lambda p: SGD(p, 0.3)), ("adam", lambda p: Adam(p, 0.1))]:
            model = MLP([2, 12, 12, 1], seed=3)
            fit(model, xs, ys, log_loss, epochs=300, optimizer=make(model.parameters()))
            scores[name] = accuracy(model, xs, ys)
        self.assertLess(scores["sgd"], 0.75)
        self.assertGreater(scores["adam"], 0.9)


class TestDigits(unittest.TestCase):
    def test_bitmaps_are_ten_different_35_pixel_digits(self):
        from firstgrad.data import DIGIT_BITMAPS

        self.assertEqual(len(DIGIT_BITMAPS), 10)
        self.assertTrue(all(len(b) == 35 for b in DIGIT_BITMAPS))
        self.assertEqual(len({tuple(b) for b in DIGIT_BITMAPS}), 10)

    def test_softmax_chances_add_to_one(self):
        from firstgrad.train import softmax
        from firstgrad.value import Value

        chances = softmax([Value(1.0), Value(2.0), Value(-3.0)])
        self.assertAlmostEqual(sum(c.data for c in chances), 1.0)

    def test_cross_entropy_gradient_matches_brute_force(self):
        from firstgrad.train import cross_entropy

        model = MLP([3, 4, 3], seed=7)
        x, y = [0.2, -0.5, 0.9], 2
        loss = cross_entropy([model(x)], [y])
        model.zero_grad()
        loss.backward()
        eps = 1e-6
        for p in model.parameters():
            claimed, original = p.grad, p.data
            p.data = original + eps
            up = cross_entropy([model(x)], [y]).data
            p.data = original - eps
            down = cross_entropy([model(x)], [y]).data
            p.data = original
            self.assertAlmostEqual(claimed, (up - down) / (2 * eps), places=5)

    def test_reads_unseen_noisy_digits(self):
        from firstgrad.data import noisy_digits
        from firstgrad.optim import Adam
        from firstgrad.train import class_accuracy, cross_entropy

        xs, ys = noisy_digits(copies=8, flip=0.08, seed=0)
        tx, ty = noisy_digits(copies=20, flip=0.08, seed=1)
        model = MLP([35, 16, 10], seed=2)
        fit(model, xs, ys, cross_entropy, epochs=100, batch_size=20,
            optimizer=Adam(model.parameters(), lr=0.05))
        self.assertGreater(class_accuracy(model, tx, ty), 0.85)
        # guessing among ten would score 10%
