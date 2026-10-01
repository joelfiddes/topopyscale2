# Contributing to TopoPyScale 2

Public releases of TopoPyScale 2 are **static snapshots** of a private development repository.
A pull request against a snapshot cannot be merged: the next release is exported from the
development repository and would overwrite it.

**Contributions come in as issues, and they are very welcome.**

- **Bug reports** ("Bug report" form): what you ran, what you expected, what happened, with
  your `config.yaml`, the full error, and `tps2 --version`.
- **Feature requests** ("Feature request" form): the problem you are trying to solve, and how
  you would know it is solved.
- **Results that disagree with your observations:** the numbers, where they came from, and the
  place, elevation and period. Check [known limitations](docs/known_limitations.md) first; many
  are already measured there.

Drafting an issue with an AI assistant is fine and often makes a better report. Point it at
[`AGENTS.md`](AGENTS.md), which describes the code and the issue format, and read what it writes
before you submit.

**Have a fix?** Describe it in the issue, with a patch or a snippet if that is clearest. It is
applied in the development repository and credited in the release notes.

TPS2 is MIT-licensed and developed by Joel Fiddes, [Mountain Futures](https://mountainfutures.ch).
If you use it in research, please cite it ([`CITATION.cff`](CITATION.cff)); if TPS2 is central
to your work, consider getting in touch about co-authorship.
