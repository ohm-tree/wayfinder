"""
agent.py

This module contains the Game class, the abstract base class for any one-player,
perfect information, abstract strategy game.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Hashable, Iterator, TypeVar, list

import numpy as np
from game import Game, State

MoveType = TypeVar('MoveType', bound=Hashable)

StateType = TypeVar('State', bound=State)
GameType = TypeVar('GameType', bound=Game[MoveType, StateType])


class Agent(Generic[GameType, StateType, MoveType],
            ABC):

    def __init__(self, game: GameType):
        self.game = game

    @abstractmethod
    async def policy(self, state: StateType) -> np.ndarray:
        """
        Returns the policy for the game state.
        """
        raise NotImplementedError

    @abstractmethod
    async def value(self, state: StateType) -> float:
        """
        Returns the value for the game state.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_active_move(self, state: StateType, index: int) -> MoveType:
        """
        Returns the active move at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    async def active_moves(self, state: StateType) -> list[MoveType]:
        """
        Returns the active moves for the game state.

        All active moves MUST have distinct hashes!!
        """
        raise NotImplementedError

    @abstractmethod
    async def index_active_move(self, state: StateType, move: MoveType) -> int:
        """
        Returns the index of the active move in the game state.
        """
        raise NotImplementedError

    @abstractmethod
    async def len_active_moves(self, state: StateType) -> int:
        """
        Returns the number of active moves in the game state.
        """
        raise NotImplementedError

    @abstractmethod
    async def require_new_move(self, state: StateType, num_moves: int) -> None:
        """
        Demand to increase the number of available moves.
        """
        raise NotImplementedError

    @abstractmethod
    async def max_moves(self, state: StateType) -> int:
        """
        Returns the maximum number of moves.
        """
        raise NotImplementedError

    @abstractmethod
    async def clear_cache(self, state: StateType) -> None:
        """
        Clears the cache of this game.
        """
        raise NotImplementedError
