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

## Layout

- `firstgrad/value.py`: one number that remembers how it was made
- `firstgrad/nn.py`: neuron, layer, network
- `firstgrad/train.py`: loss, nudge step, training loop
- `firstgrad/optim.py`: plain, momentum and Adam nudging
- `firstgrad/gradcheck.py`: the slow check that proves the fast one
