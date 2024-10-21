"""
game.py

This module contains the Game class, the abstract base class for any one-player,
perfect information, abstract strategy game.
- These games are **complex**. They require nontrivial computation to simulate, which may benefit from running concurrently and/or batching. Imagine games involving querying formal mathematical verifiers or games involving running generated code.
- These games are **large**. Moves may form a continuum, or be intractable to enumerate as in arbitrary text generation. The branching factor for these games makes them intractable for a vanilla MCTS implementation.

See game.md for more information.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Hashable, Iterator, TypeVar, list

import numpy as np


class State(ABC, Hashable):
    """
    The State class is an abstract base class
    for representing the state of a game.

    """

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def saves(cls, states: list['State'], filename: str) -> None:
        """
        Save a collection of states to a file.

        This is super important. During MCTS,
        this is what gets saved to the replay buffer.

        Parameters
        ----------
        states : list[State]
            A list of game states to save.
        filename : str
            The name of the file to save the states to.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def loads(cls, filename: str) -> list['State']:
        """
        Load a collection of states from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the states from.
        """
        raise NotImplementedError


MoveType = TypeVar('MoveType', bound=Hashable)

StateType = TypeVar("StateType", bound=State)


class Game(Generic[MoveType, StateType], ABC, Hashable):
    """
    The Game class is an abstract base class
    for representing a one-player, perfect information,
    abstract strategy game.

    This class should be subclassed, and the un-underscored
    methods should be implemented. The underscored methods
    cache the final results of the un-underscored methods,
    and should not be overwritten.

    """

    def __init__(self):
        self.starting_state_cache = {}
        self.is_legal_cache = {}
        self.next_state_cache = {}
        self.terminal_cache = {}
        self.reward_cache = {}
        self.victorious_cache = {}

    @property
    @abstractmethod
    def death_value(self):
        raise NotImplementedError

    # Cached versions
    async def _starting_state(self, *args, **kwargs) -> StateType:
        if hash((args, kwargs)) not in self.starting_state_cache:
            self.starting_state_cache[hash((args, kwargs))] = await self.starting_state(*args, **kwargs)
        return self.starting_state_cache[hash((args, kwargs))]

    async def _is_legal(self, state: StateType, action: MoveType) -> bool:
        if hash((state, action)) not in self.is_legal_cache:
            self.is_legal_cache[hash((state, action))] = await self.is_legal(state, action)
        return self.is_legal_cache[hash((state, action))]

    async def _next_state(self, state: StateType, action: MoveType) -> StateType:
        if hash((state, action)) not in self.next_state_cache:
            self.next_state_cache[hash((state, action))] = await self.next_state(state, action)
        return self.next_state_cache[hash((state, action))]

    async def _terminal(self, state: StateType) -> bool:
        if hash(state) not in self.terminal_cache:
            self.terminal_cache[hash(state)] = await self.terminal(state)
        return self.terminal_cache[hash(state)]

    async def _reward(self, state: StateType) -> float:
        if hash(state) not in self.reward_cache:
            self.reward_cache[hash(state)] = await self.reward(state)
        return self.reward_cache[hash(state)]

    async def _victorious(self, state: StateType) -> bool:
        if hash(state) not in self.victorious_cache:
            self.victorious_cache[hash(state)] = await self.victorious(state)
        return self.victorious_cache[hash(state)]

    async def clear_cache(self) -> None:
        """
        Clears the cache of this game.

        If you overwrite this method, call the parent method
        at the end of your implementation.
        """
        self.starting_state_cache = {}
        self.is_legal_cache = {}
        self.next_state_cache = {}
        self.terminal_cache = {}
        self.reward_cache = {}
        self.victorious_cache = {}

    # Un-cached versions, for users to implement.

    @abstractmethod
    async def starting_state(self, *args, **kwargs) -> StateType:
        """
        Returns the starting state of the game.
        """
        raise NotImplementedError

    @abstractmethod
    async def is_legal(self, state: StateType, action: MoveType) -> bool:
        """
        Returns True if the action is legal, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def next_state(self, state: StateType, action: MoveType) -> StateType:
        """
        Returns the next state of the game given a current state and action.
        Requires that the state is non-terminal and the action is legal.
        """
        raise NotImplementedError

    @abstractmethod
    async def terminal(self, state: StateType) -> bool:
        """
        Returns True if the game is over, False otherwise.

        Convention: if a state is non-terminal, then there must exist at least one legal move!
        There are two ways to handle this:
        1. Make all moves legal.
         - Do not raise any errors for illegal moves in next_state
         - Do legality checking in the terminal() function applied to the *subsequent* state.
        2. In terminal(), perform a check for whether there exist any legal moves.
        """
        raise NotImplementedError

    @abstractmethod
    async def reward(self, state: StateType) -> float:
        """
        Returns a float consisting of the reward for the player.
        """
        raise NotImplementedError

    @abstractmethod
    async def victorious(self, state: StateType) -> bool:
        """
        Returns True if the player has won, False otherwise.

        This is useful for games in which your search can terminate
        early if you know you have won.
        """
        raise NotImplementedError

    @abstractmethod
    async def make_root(self, state: StateType) -> None:
        """
        Severs any references to the parent of
        this state, making it the root of the tree.
        """
        raise NotImplementedError
