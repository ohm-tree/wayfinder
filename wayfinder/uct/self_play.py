"""
self_play.py

This module contains the self-play function.
This returns the game states and action distributions, as well as the final result.

"""

import asyncio
import logging
from importlib.metadata import distribution
from math import dist
from typing import Any, Generic, Hashable, Iterator, Optional, TypedDict, TypeVar

import numpy as np

from wayfinder.games import *
from wayfinder.uct.tree_diagnostics import TreeDiagnostics
from wayfinder.uct.uct_alg import async_uct_search
from wayfinder.uct.uct_node import UCTNode

MoveType = TypeVar('MoveType', bound=Hashable)
StateType = TypeVar('State', bound=State)
GameType = TypeVar('GameType', bound=Game[MoveType, StateType])
AgentType = TypeVar('AgentType', bound=Agent[GameType, StateType, MoveType])


class SelfPlayResult(TypedDict):
    states: list[Any]
    distributions: list[np.ndarray]
    rewards: list[float]
    result: float
    tree_diagnostics: Optional[list[str]]


async def async_self_play(
    logger: logging.Logger,
    state: StateType,
    game: GameType,
    agent: AgentType,
    # Dopamine, but may incur some memory.
    return_tree_diagnostics: bool = True,
    tree_kwargs: dict[str, Any] = {},
    search_kwargs: dict[str, Any] = {},
) -> SelfPlayResult:
    """
    Main asynchronous outer-loop for self-play.

    Each iteration, runs a full tree search, then selects an action based on the visit distribution
    and permanently plays the action in the game.

    Parameters:
    ------------
    logger: logging.Logger
        Logger for logging information.
    state: StateType
        The initial state of the game.
    game: GameType
        The game instance.
    agent: AgentType
        The agent instance.
    return_tree_diagnostics: bool
        Whether to return tree diagnostics.
    tree_kwargs: dict[str, Any]
        Additional arguments for the tree, passed into UCTNode constructor.
    search_kwargs: dict[str, Any]
        Additional arguments for the search, passed into async_uct_search.
    """

    states: list[Any] = []
    distributions: list[np.ndarray] = []

    # Send those in.
    root = UCTNode(
        agent=agent,
        game=game,
        state=state,
        action_idx=-1,
        parent=None,
        **tree_kwargs
    )

    if return_tree_diagnostics:
        tree_diagnostics = []
        diagnostics = TreeDiagnostics(
            game=game
        )

    states.append(root.state)

    move_count = 0

    while not (await root.terminal()):
        logger.info("Move: " + str(move_count))
        move_count += 1
        logger.info(root.state.__str__())
        """
        TODO: Fast Playouts would be implemented here.
        """
        winning_node: UCTNode
        res = await async_uct_search(
            logger=logger,
            root=root,
            **search_kwargs
        )
        active_moves = res["active_moves"]
        distribution = res["visit_distribution"]
        winning_node = res["winning_node"]

        # TODO: more configurations possible for uct_search, not used right now.

        """
        TODO: In MCTS algorithms, people sometimes change up the temperature right here,
        to sharpen the training distribution. This is something we could try.
        """
        if return_tree_diagnostics:
            tree_diagnostics.append(diagnostics.tree_to_string(root))

        if winning_node is not None:
            root = winning_node
            states.append(winning_node.state)
            break

        distributions.append(distribution)
        logger.info(f"Action distribution: {distribution}")

        action_idx = np.random.choice(len(distribution), p=distribution)

        root = root.children[action_idx]
        # set root parent to None so that it knows it is the root.
        await root.root()
        states.append(root.state)

    logger.info("Move: " + str(move_count))
    logger.info(root.state.__str__())

    # The reward for all states in the tree is the reward of the final state.

    final_reward = await root.reward()
    if winning_node is not None:
        logger.info(
            "Game finished early with reward: " + str(final_reward))
    else:
        logger.info(
            f"Game finished after {move_count} moves with reward: {final_reward}")
    rewards = [final_reward for _ in states]
    return {
        "states": states,
        "distributions": distributions,
        "rewards": rewards,
        "result": final_reward,
        "tree_diagnostics": tree_diagnostics if return_tree_diagnostics else None
    }


def self_play(
    logger: logging.Logger,
    state: StateType,
    game: GameType,
    agent: AgentType,
    return_tree_diagnostics: bool = True,
    tree_kwargs: dict[str, Any] = {},
    search_kwargs: dict[str, Any] = {},
) -> SelfPlayResult:
    """
    Syncronous wrapper for async_self_play.

    (We do not usually use this, because our projects are usually
    set up to make asynchronous process-external queries, hence there
    is already global asyncio loop running external to async_self_play).

    See async_self_play for details.
    """
    return asyncio.run(
        async_self_play(
            logger,
            state,
            game,
            agent,
            return_tree_diagnostics,
            tree_kwargs,
            search_kwargs
        )
    )
