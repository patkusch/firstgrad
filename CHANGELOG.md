# Changelog

## November 2026
- Training can now pull every number a little towards zero after each nudge (weight decay), and can stop when the loss on held-out examples stops falling, going back to its best moment. Both are there to stop the network memorising. The name-writer has not been rerun with them yet.
- Housekeeping: a security policy, a citation file, a contributing guide, weekly Dependabot, and a tidier CI run.

## October 2026
- The gradient engine, proved against a slow brute-force check.
- Networks, a training loop, and the XOR and spiral examples.
- Momentum and Adam. The hard spirals went from 57% to 92% right on points never seen in training.
- Reading noisy hand-drawn digits: 94% right with 10% static.
- A name-writer that learns which letter follows which, scored against plain counting on names it never saw (2.69 against 2.73, lower is better: a tie, as expected with one letter of memory).
- Letting the name-writer look two or three letters back made it worse on new names (3.77 at three back, against 3.30 for blind guessing): it memorised the training names instead of learning how names work.
