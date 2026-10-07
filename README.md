# firstgrad

A neural network library small enough to read in one sitting, with no ML
libraries underneath. It exists to answer one question: how does a model learn?

## What it does

A model is a pile of numbers. Learning means nudging those numbers so the
model's answers get less wrong. The hard part is knowing which way to nudge
each one. `firstgrad` works that out for you and shows its working.

## Why you can trust it

Every nudge direction is checked against a slow, obviously correct method:
move the number a tiny bit each way and watch what happens. The tests run that
comparison on every operation. If the clever method and the slow method
disagree, the build fails.

## Run the tests

```bash
python3 -m unittest discover -s tests -t .
```

## Try it learn something

```bash
python3 -m examples.xor        # a problem one neuron cannot solve; 4/4 right
python3 -m examples.spirals    # two interleaved spirals; 98% right, with a map
```

## What went wrong, and the fix

The first spiral test used three full turns and got stuck at 62% right (barely
better than a coin flip) with plain nudging. Easing it to one and a half turns
worked, but that dodged the problem. So the smarter nudging methods in
`firstgrad/optim.py` went back at the hard version:

| Method | Right on spirals it trained on | Right on fresh spirals |
|---|---|---|
| Plain | 63% | 57% |
| Momentum (keeps some of its last direction) | 97% | 92% |
| Adam (gives each number its own step size) | 100% | 92% |

Same network, same data, same number of rounds. Only the nudging changed. The
last column matters most: those spirals were never seen in training, so a high
score there means it learned the shape rather than memorising the points. Adam
getting 100% on the old ones but 92% on new ones shows a little memorising.

```bash
python3 -m examples.hard_spirals
```

## Reading handwritten digits

Ten digits drawn by hand on a 5-by-7 grid. The network trains on copies where
8% of the pixels are randomly flipped, like static on a screen, then is scored
on fresh copies it has never seen. Guessing among ten digits would get 10%.

| Static on the fresh digits | Right |
|---|---|
| 0% | 100% |
| 5% | 98% |
| 10% | 94% |
| 15% | 90% |
| 25% | 66% |
| 35% | 37% |

It was trained on 8% static, so it holds up well at that level and fades as
the picture gets worse than anything it practised on. At 15% static the
mix-ups are mostly the 8 being read as something else, which makes sense: an
8 shares pixels with almost every other digit.

```bash
python3 -m examples.digits
```

## Writing made-up names

Given one letter, guess the next one. Do that over and over and you have a
name-writer. It is tried two ways: counting which letter follows which in 68
real names, and a network that learns the same thing. Both are scored on 69
other names they never saw. The score is how surprised the model is by the real
next letters, so lower is better, and blind guessing scores 3.30.

| Method | Surprise on unseen names |
|---|---|
| Blind guessing | 3.30 |
| Counting | 2.73 |
| Network | 2.69 |

The network ties with counting, which is the expected result: with only one
letter of memory there is nothing more to learn than the counts. The names
reflect that. Counting gives things like "fdqana" and "adricmicsogontq", the
network gives "ell", "jak" and "adranahe". Closer to real names, still not good.
An untrained network scores 4.40, worse than blind guessing, because its random
starting numbers make it confidently wrong.

```bash
python3 -m examples.names
```

The next step is letting it look two or three letters back. Counting cannot
follow, because the table grows too big to fill from 137 names. A network can.

## Layout

- `firstgrad/value.py`: one number that remembers how it was made
- `firstgrad/nn.py`: neuron, layer, network
- `firstgrad/train.py`: loss, nudge step, training loop
- `firstgrad/data.py`: XOR, spirals, hand-drawn digits with static
- `firstgrad/text.py`: letters as numbers, counting baseline, name-writer
- `firstgrad/optim.py`: plain, momentum and Adam nudging
- `firstgrad/gradcheck.py`: the slow check that proves the fast one
