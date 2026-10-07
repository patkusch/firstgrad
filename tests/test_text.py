import math
import random
import unittest

from firstgrad.data import NAMES
from firstgrad.text import INDEX, LETTERS, average_surprise, count_table, pairs, write_name


class TestCounting(unittest.TestCase):
    def test_pairs_include_start_and_end(self):
        self.assertEqual(pairs(["ab"]), [(INDEX["."], INDEX["a"]), (INDEX["a"], INDEX["b"]), (INDEX["b"], INDEX["."])])

    def test_each_row_of_chances_adds_to_one(self):
        for row in count_table(NAMES):
            self.assertAlmostEqual(sum(row), 1.0)

    def test_counting_beats_random_guessing(self):
        train, held_out = NAMES[::2], NAMES[1::2]
        score = average_surprise(count_table(train), held_out)
        self.assertLess(score, math.log(len(LETTERS)) - 0.3)

    def test_names_use_only_known_letters(self):
        self.assertTrue(all(ch in INDEX for name in NAMES for ch in name))

    def test_written_names_are_made_of_letters_and_finish(self):
        table = count_table(NAMES)
        rng = random.Random(0)
        for _ in range(20):
            name = write_name(lambda a: table[a], rng)
            self.assertTrue(name.isalpha() or name == "")


if __name__ == "__main__":
    unittest.main()


class TestNetworkWriter(unittest.TestCase):
    def train(self, names, epochs):
        from firstgrad.nn import MLP
        from firstgrad.optim import Adam
        from firstgrad.text import one_hot
        from firstgrad.train import cross_entropy, fit

        ps = pairs(names)
        xs, ys = [one_hot(a) for a, _ in ps], [b for _, b in ps]
        model = MLP([len(LETTERS), 20, len(LETTERS)], seed=4)
        fit(model, xs, ys, cross_entropy, epochs=epochs, batch_size=64,
            optimizer=Adam(model.parameters(), lr=0.05))
        return model

    def test_network_learns_about_as_well_as_counting(self):
        from firstgrad.text import network_surprise

        train, held_out = NAMES[::2], NAMES[1::2]
        model = self.train(train, epochs=100)
        counted = average_surprise(count_table(train), held_out)
        learned = network_surprise(model, held_out)
        from firstgrad.nn import MLP

        untrained = network_surprise(MLP([27, 20, 27], seed=4), held_out)
        self.assertGreater(untrained, learned + 1.0)
        self.assertLess(learned, math.log(27) - 0.3)
        self.assertLess(abs(learned - counted), 0.25)
