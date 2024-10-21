import asyncio
import random
from typing import Optional

import numpy as np

from src.games.game import Game, State


class SudokuState(State):
    def __init__(self, board: np.ndarray, dead: bool = False):
        self.board = board
        self.dead = False

    @classmethod
    def saves(cls, states: list['SudokuState'], filename: str) -> None:
        """
        Save a collection of states to a file.
        """
        res = np.array([state.board for state in states])
        np.save(filename, res)

    @classmethod
    def loads(cls, filename: str) -> list['SudokuState']:
        """
        Load a collection of states from a file.
        """
        res = np.load(filename)
        return [SudokuState(board=board) for board
                in res]

    @classmethod
    def from_string(cls, s: str) -> 'SudokuState':
        """
        Create a SudokuGameState from a string representation of the board,
        which is a 81-length string of '0' through '9,' where '0' represents an empty cell.
        """
        board = np.array([int(c) for c in s]).reshape((9, 9))
        return SudokuState(board)

    def to_numpy(self) -> np.ndarray:
        """
        Convert the game state to a tensor (n, r, c).
        game.state.board is currently a numpy array of size (r, c) = (9, 9) with values from 0 to 9.
        This should be one-hot encoded to shape (9, 9, 9).
        The one hot takes in a long tensor, so we need to convert it to a tensor with dtype long.
        Then, one_hot will create a tensor of shape (r, c, n) = (9, 9, 10), we permute it to (10, 9, 9),
        and slice it to (9, 9, 9).
        Finally, we convert it to float because one_hot returns a tensor of dtype long.
        """
        one_hot = np.zeros((9, 9, 10), dtype=int)  # (r, c, n + 1)
        one_hot[np.arange(9)[:, None], np.arange(9), self.board] = 1
        one_hot = one_hot[:, :, 1:].astype(float)  # (r, c, n)
        one_hot = np.transpose(one_hot, (2, 0, 1))  # (n, r, c)
        return one_hot

    def __str__(self):
        return '\n'.join(' '.join(str(cell) if cell != 0 else '.' for cell in row) for row in self.board)

    def __hash__(self):
        return hash(self.board.tobytes())


class SudokuMove:
    def __init__(self, row: int, col: int, number: int):
        self.row = row
        self.col = col
        self.number = number

    def __str__(self):
        return f"({self.row}, {self.col}, {self.number})"


class SudokuGame(Game[SudokuMove, SudokuState]):
    """
    A Sudoku game implementation of the Game class.

    This is used for testing the MCTS algorithm.
    """

    def __init__(self,
                 timeout_jitter: float = 0.1,
                 legality_on_move: bool = True
                 ):
        super().__init__()
        self.timeout_jitter = timeout_jitter
        self.legality_on_move = legality_on_move

    async def starting_state(self, board: Optional[np.ndarray] = None) -> SudokuState:
        """
        Returns a new Sudoku game state.
        """

        await asyncio.sleep(random.random() * self.timeout_jitter)
        if board is None:
            return SudokuState(board=np.zeros((9, 9), dtype=int))
        return SudokuState(board=board)

    async def is_legal(self, state: SudokuState, action: SudokuMove) -> bool:
        """
        If legality_on_move is False, this method always returns True.
        Otherwise, it checks if the move is legal on the board.
        """
        if self.legality_on_move:
            return await self.is_actually_legal(state, action)
        return True

    async def is_actually_legal(self, state: SudokuState, action: SudokuMove) -> bool:
        """
        Determines if placing a number in the given row and column is legal, assuming the move just made.
        """
        await asyncio.sleep(random.random() * self.timeout_jitter)
        row, col, number = action.row, action.col, action.number
        # Check if the position was already filled
        if state.board[row, col] != 0:
            return False

        # Check row, column, and 3x3 grid for the number
        if number in state.board[row] or number in state.board[:, col]:
            return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if number in state.board[start_row:start_row+3, start_col:start_col+3]:
            return False
        return True

    async def next_state(self, state: SudokuState, action: SudokuMove) -> SudokuState:
        """
        Places a number on the Sudoku board.
        Action is a tuple: (row, col, number)
        Checks if the move made is valid, if not, marks the state as dead.
        """
        await asyncio.sleep(random.random() * self.timeout_jitter)
        if self.legality_on_move:
            assert await self.is_actually_legal(state, action), "Illegal move attempted."

        row, col, number = action.row, action.col, action.number
        new_board = np.copy(state.board)
        new_board[row, col] = number

        # Check if the move is legal only for the new move
        if (not self.legality_on_move) and (not self.is_actually_legal(state, row, col, number)):
            return SudokuState(board=new_board, is_dead=True)

        return SudokuState(board=new_board)

    async def terminal(self, state: SudokuState) -> bool:
        """
        The game is considered over if the board is fully filled or if the state is dead.
        """
        await asyncio.sleep(random.random() * self.timeout_jitter)
        if state.dead:
            return True
        return np.all(state.board != 0)

    async def reward(self, state: SudokuState) -> float:
        """
        Rewards the player if the board is correctly completed, otherwise 0.
        """
        await asyncio.sleep(random.random() * self.timeout_jitter)
        assert self.terminal(
            state), "Reward can only be calculated for terminal states."
        if state.dead:
            return -1.0

        assert self.is_board_valid(
            state.board), "Board is not valid, but reached terminal non-dead state."
        return 1.0

    async def victorious(self, state: SudokuState) -> bool:
        """
        Returns True if the board is correctly completed.
        """
        await asyncio.sleep(random.random() * self.timeout_jitter)
        return self.reward(state) == 1.0

    async def make_root(self, state: SudokuState) -> None:
        """
        No need to do anything here.
        """
        pass

    def is_board_valid(self, board: np.ndarray) -> bool:
        """
        Check if the entire board is valid. This never needs to be called,
        because we check legality on every move, so at all times the board
        is valid.
        """
        for i in range(9):
            if not self.is_group_valid(board[i]) or not self.is_group_valid(board[:, i]):
                return False
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                if not self.is_group_valid(board[row:row+3, col:col+3].flatten()):
                    return False
        return True

    def is_group_valid(self, group: np.ndarray) -> bool:
        """
        Checks if a group (row, column, or block) contains no duplicates of numbers 1-9.
        """
        return np.all(np.bincount(group, minlength=10)[1:] <= 1)
