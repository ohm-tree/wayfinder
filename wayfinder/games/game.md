This module contains the Game class, the abstract base class for any one-player,
perfect information, abstract strategy game.
- These games are **complex**. They require nontrivial computation to simulate, which may benefit from running concurrently and/or batching. Imagine games involving querying formal mathematical verifiers or games involving running generated code.
- These games are **large**. Moves may form a continuum, or be intractable to enumerate as in arbitrary text generation. The branching factor for these games makes them intractable for a vanilla MCTS implementation.


Design principle for these super-core things:
 - A State() should be a data-only class. The only methods it is allowed to have are data-related,
 such as saving and loading.
 - A Game() should have a reference to a Worker() which can query other processes for difficult-to-compute
 state transitions, legality checks, and rewards.

There is now a design decision to make at this stage, because whenever we compute things like
state transitions, legality checks, and rewards, we would love to cache these values so that we don't
have to recompute them every time. So the decision is where to put them; should we put them in
the State, and increase the size of the State, or put them in the Game.

In this project, we will take the perspective that for any classes that the user has to implement,
*data classes should be light* and *agents should be heavy*.
In this case, the Game is the more agent-like object, and should maintain caches which
are e.g. dict[State, dict[Move, State]] and dict[State, float] for state transitions and rewards.

So the workflow now looks like, because States are so light, you should feel free to pass them
around all your processes etc. Whenever you need to query something like the reward or game transitions,
you ask the corresponding Agent which has responsibility to either compute or query the cache.

In previous implementations, we had things like MetaGameStates which contained references
to a GameState in order to extend the GameState's functionality. This is a lot of references,
and I have decided that instead we should have "just one reference" from a "Meta Game Agent"
to a "Game Agent," which the "Meta Game Agent" can then use to query the "Game Agent" for
functionality.

This now leads to a different issue which is that of object lifetime. When a State object (which
will end up being acted on by many agents) is destroyed, it should also destroy the cached
values in the Game object. We take the perspective that this would lead to significant amounts of
bookkeeping, and so we will not do this. Instead, agentic objects like the Game should expose a
method like `clear_cache()` which will clear the cache entirely, such that new queries will be
computed from scratch.

Note that none of this is applicable to the UCTNode, which is something we'll
write and the user won't need to touch. That thing has a lot more mixing of responsibilties.