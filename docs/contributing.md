# Contributing

TopoPyScale 2 is developed in a private repository and published as **static snapshots**.
Each release is exported from the development repository, so a pull request against a
snapshot cannot be merged back: the next release would overwrite it.

**Contributions come in as GitHub issues**, and they are very welcome:

- **Bug reports:** what you ran, what you expected, what happened. Include your
  `config.yaml`, the output of `tps2 info --config config.yaml`, your TPS2 version
  (`tps2 --version`) and the full error.
- **Feature requests:** the problem you are trying to solve, not only the solution you
  have in mind, and how you would know it works.
- **Science questions and results:** where TPS2 disagrees with your observations is exactly
  what the [Known limitations](known_limitations.md) page is built from.

The issue forms ask for these. Drafting an issue with an AI assistant is fine, and often
makes a better report: point it at `AGENTS.md` in the repository, which explains the codebase
and the issue format. Please read what it writes before you submit it.

## Code

If you have a fix, describe it in the issue, with a patch or a snippet if that is the
clearest way. It will be applied in the development repository and credited in the release
notes.

## Citing and co-authorship

TPS2 is open source under the MIT licence, developed by Joel Fiddes at
[Mountain Futures](https://mountainfutures.ch) and Simon Filhol at Météo-France. If you use it in published work, please cite
it (`CITATION.cff` in the repository). If TPS2 is central to your work, consider getting in
touch about co-authorship or collaboration.
