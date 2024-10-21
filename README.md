# Wayfinder

Wayfinder is an MCTS implementation a la [AlphaGo](https://arxiv.org/abs/1712.01815). It plays large one-player, perfect-information, abstract strategy games.

### Objectives

- We target **complex games**. They require nontrivial computation to simulate, which may benefit from running concurrently and/or batching. Imagine games involving querying formal mathematical verifiers or games involving running generated code.
- We target **large games**. Moves may form a continuum, or be intractable to enumerate as in arbitrary text generation. The branching factor for these games makes them intractable for a vanilla MCTS implementation.
- We target **complex agents**. Imagine agents which are querying deep models.

### Development

If you want to install in editable mode locally, my understanding is that due to (recent changes in pip)[https://github.com/mne-tools/mne-python/issues/12169] you need to run the following command instead of `pip install -e .`:

```bash
pip install --config-settings editable-mode=strict -e .
```