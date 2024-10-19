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

MoveType = TypeVar('MoveType', bound=Hashable)


class State(Generic[MoveType], ABC, Hashable):
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


StateType = TypeVar("StateType", bound=State)


class Game(Generic[MoveType, StateType], ABC, Hashable):

    @classmethod
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
        Requires that the state is non-terminal and action is legal.
        """
        raise NotImplementedError

    @abstractmethod
    async def terminal(self, state: StateType) -> bool:
        """
        Returns True if the game is over, False otherwise.
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

    @abstractmethod
    async def clear_cache(self, state: StateType) -> None:
        """
        Clears the cache of this game.
        """
        raise NotImplementedError
