"""
agent.py

This module contains the Game class, the abstract base class for any one-player,
perfect information, abstract strategy game.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Hashable, Iterator, Optional, TypeVar

import numpy as np

from wayfinder.games.game import Game, State

MoveType = TypeVar('MoveType', bound=Hashable)

StateType = TypeVar('State', bound=State)
GameType = TypeVar('GameType', bound=Game[MoveType, StateType])


class Agent(Generic[GameType, StateType, MoveType],
            ABC):

    def __init__(self, game: GameType):
        self.game = game
        self.policy_cache = {}
        self.value_cache = {}
        self.index_active_move_cache = {}

    async def _policy(self, state: StateType, move: MoveType) -> np.ndarray:
        """
        Returns the policy for the game state. Cached.
        """
        if hash((state, move)) not in self.policy_cache:
            self.policy_cache[hash((state, move))] = await self.policy(state, move)
        return self.policy_cache[hash((state, move))]

    @abstractmethod
    async def policy(self, state: StateType, move: MoveType) -> np.ndarray:
        """
        Returns the policy for the game state.
        """
        raise NotImplementedError

    async def _value(self, state: StateType) -> float:
        """
        Returns the value for the game state. Cached.
        """
        if hash(state) not in self.value_cache:
            self.value_cache[hash(state)] = await self.value(state)
        return self.value_cache[hash(state)]

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
        Not cached; user should provide own caching implementation.
        """
        raise NotImplementedError

    @abstractmethod
    async def active_moves(self, state: StateType) -> list[MoveType]:
        """
        Returns the active moves for the game state.

        All active moves must have distinct hashes, and be legal!

        Not cached; user should provide own caching implementation.
        """
        raise NotImplementedError

    async def _index_active_move(self, state: StateType, move: MoveType) -> int:
        """
        Returns the index of the active move in the game state.
        Cached; should not change under required_new_move.
        """
        if hash((state, move)) not in self.index_active_move_cache:
            self.index_active_move_cache[hash((state, move))] = await self.index_active_move(state, move)
        return self.index_active_move_cache[hash((state, move))]

    @abstractmethod
    async def index_active_move(self, state: StateType, move: MoveType) -> int:
        """
        Returns the index of the active move in the game state.
        Not cached; user should provide own caching implementation.
        """
        raise NotImplementedError

    @abstractmethod
    async def len_active_moves(self, state: StateType) -> int:
        """
        Returns the number of active moves in the game state.
        Not cached; user should provide own caching implementation.
        """
        raise NotImplementedError

    @abstractmethod
    async def require_new_move(
        self,
        state: StateType,
        min_num_moves: int,
        max_num_moves: Optional[int] = None
    ) -> bool:
        """
        Demand to increase the number of available moves. state must be a non-terminal node.

        Part of the contract for Game implementations is that any non-terminal
        node must have at least one available move. However, it may be that this move is
        hard to find, etc. To this end, require_new_move is allowed to create
        any number of moves in the range [min_num_moves, max_num_moves], inclusive.

        If the agent is unable to make any legal move at all, then it should return False.
        This indicates that the node should actually be marked terminal, and will be handled
        as a special case in the UCT algorithm.
        """
        raise NotImplementedError

    @abstractmethod
    async def max_moves(self, state: StateType) -> int:
        """
        Returns the maximum number of moves; the MCTS will not
        attempt to request more moves than this. This should
        not be 0.

        Not cached.
        """
        raise NotImplementedError

    async def clear_cache(self) -> None:
        """
        Clears the cache of this agent.

        If you overwrite this method, call the parent method
        at the end of your implementation.
        """
        self.policy_cache = {}
        self.value_cache = {}
        self.index_active_move_cache = {}
