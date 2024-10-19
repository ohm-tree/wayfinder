# Wayfinder

Wayfinder is an MCTS implementation a la [AlphaGo](https://arxiv.org/abs/1712.01815). It plays large one-player, perfect-information, abstract strategy games.

### Objectives

- We target **complex games**. They require nontrivial computation to simulate, which may benefit from running concurrently and/or batching. Imagine games involving querying formal mathematical verifiers or games involving running generated code.
- We target **large games**. Moves may form a continuum, or be intractable to enumerate as in arbitrary text generation. The branching factor for these games makes them intractable for a vanilla MCTS implementation.
- We target **complex agents**. Imagine agents which are querying deep models.