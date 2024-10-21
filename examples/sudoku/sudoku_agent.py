import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np

from examples.sudoku.sudoku import SudokuGame, SudokuMove, SudokuState
from examples.workers.worker import Worker
from src.games.agent import Agent


class SudokuAgent(Agent[SudokuGame, SudokuState, SudokuMove]):
    """
    This sudoku agent is mostly for testing purpose.
    """

    def __init__(self,
                 game: SudokuGame,
                 worker: Worker,
                 ):
        super().__init__(game=game)
        self.worker = worker

        self.full_result_cache: dict[SudokuState, dict[str, np.ndarray]] = {}

        self._active_moves: dict[SudokuState, list[SudokuMove]] = {}
        self.max_moves: dict[SudokuState, int] = {}
        self.exposed: dict[SudokuState, int] = {}

    async def compute(self, state: SudokuState):
        if hash(state) in self.full_result_cache:
            return
        result = await self.worker.query(
            task={
                'channel': self.worker.name,
            },
            channel='SudokuCNNWorker'
        )
        self.full_result_cache.update(
            {
                hash(state): {
                    'policy': result['policy'],
                    'value': result['value']
                }
            }
        )

    async def policy(self, state: SudokuState, move: SudokuMove) -> np.ndarray:
        await self.compute(state)
        return self.full_result_cache[hash(state)]['policy'][move.row, move.col, move.number - 1]

    async def value(self, state: SudokuState) -> float:
        await self.compute(state)
        return self.full_result_cache[hash(state)]['value']

    async def get_active_move(self, state: SudokuState, index: int) -> SudokuMove:
        if index >= self.exposed[state]:
            raise IndexError(f"Index {index} out of bounds for active moves.")

        return self._active_moves[state][index]

    async def active_moves(self, state: SudokuState) -> list[SudokuMove]:
        if state not in self._active_moves:
            # Don't add this to _active_moves keys, because we use contains to check for pre-computation.
            return []
        return self._active_moves[:self.exposed[state]]

    async def index_active_move(self, state: SudokuState, move: SudokuMove) -> int:
        index = self._active_moves[state].index(move)
        if index >= self.exposed[state]:
            raise ValueError(f"Move {move} not in active moves.")
        return index

    async def compute_active_moves(self, state: SudokuState):
        if state in self._active_moves:
            return  # Already computed.

        # In our case, we can simply rank all of the legal moves by their policy value.
        # First... we need to make an action mask.
        action_mask = np.zeros((9, 9, 9), dtype=np.bool_)
        num_legal_moves = 0
        # we'll just do this the dumb way.
        for row in range(9):
            for col in range(9):
                for num in range(1, 10):
                    if await self.game.is_legal(state, SudokuMove(row, col, num)):
                        action_mask[row, col, num - 1] = True
                        num_legal_moves += 1

        # Now, take the policy values and apply the mask.
        policy_values = self.full_result_cache[hash(state)]['policy']
        policy_values[~action_mask] = -np.inf

        # argsort the policy values.
        # This will give us the indices of the policy values in sorted order.
        sorted_indices = np.argsort(policy_values, axis=None)[::-1]

        # Now, we can just take the first moves.
        self._active_moves[state] = []
        for i in range(num_legal_moves):
            row, col, num = np.unravel_index(
                sorted_indices[i], policy_values.shape)
            self._active_moves[state].append(SudokuMove(row, col, num + 1))

    async def require_new_move(
        self,
        state: SudokuState,
        min_num_moves: int,
        max_num_moves: Optional[int] = None
    ) -> bool:
        await self.compute(state)
        await self.compute_active_moves(state)

        if len(self._active_moves[state]) == 0:
            return False  # We were unable to create any moves.
        if max_num_moves is None:
            max_num_moves = min_num_moves

        # check to make sure we have enough moves.
        if max_num_moves > self.max_moves[state]:
            raise ValueError(
                f"Requested more moves than max_moves. Requested {max_num_moves}, max is {self.max_moves[state]}.")
        self.exposed = max(self.exposed, max_num_moves)

        return True

    async def max_moves(self, state: SudokuState) -> int:
        await self.compute_active_moves(state)
        return self.max_moves[state]
