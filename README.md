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

## What went wrong along the way

The first spiral test used three full turns and got stuck at 62% right (barely
better than a coin flip) after 400 rounds of plain nudging. Easing it to one
and a half turns fixed it: 98% in 300 rounds. The lesson is that the nudging
method matters as much as the network, and the next step is to try a smarter
nudging method and see how much of the hard version it recovers.

## Layout

- `firstgrad/value.py`: one number that remembers how it was made
- `firstgrad/nn.py`: neuron, layer, network
- `firstgrad/train.py`: loss, nudge step, training loop
- `firstgrad/gradcheck.py`: the slow check that proves the fast one
