# Adding to firstgrad

Small, focused changes are welcome. Open an issue first for anything large, so nobody builds what will not be merged.

## Before you push

```bash
python3 -m unittest discover -s tests -t .
```

Push only when all of it passes. Never pipe the test output into something that hides a failure.

## What every change comes with

- A test that fails without the change and passes with it.
- No dependencies: the library stays plain Python that can be read in one sitting. The tests take about a minute, because the network is built from scratch.
- Plain English in anything a person reads: say what it does and why, and put the plain meaning next to every number.
- No invented figures. A number in the README comes from a run you can repeat, and a result that showed no benefit still goes in.
