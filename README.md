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
