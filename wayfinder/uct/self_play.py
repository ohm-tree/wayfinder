"""
self_play.py

This module contains the self-play function.
This returns the game states and action distributions, as well as the final result.

"""

import asyncio
from importlib.metadata import distribution
import logging
from math import dist
from typing import Any, Generic, Hashable, Iterator, Optional, TypedDict, TypeVar

import numpy as np

from wayfinder.games import *
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


async def async_self_play(
    logger: logging.Logger,
    state: StateType,
    game: GameType,
    agent: AgentType,
    tree_kwargs: dict[str, Any] = {},
    search_kwargs: dict[str, Any] = {},
) -> SelfPlayResult:
    """
    Play a game using a policy, and return the game states, action distributions, and final reward.
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

        if winning_node is not None:
            root = winning_node
            states.append(winning_node.state)
            break
        distributions.append(distribution)
        logger.info(f"Action distribution: {distribution}")

        action_idx = np.random.choice(len(distribution), p=distribution)

        root = root.children[action_idx]
        # set root parent to None so that it knows it is the root.
        root.root()
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
        "result": final_reward
    }


def self_play(
    logger: logging.Logger,
    state: StateType,
    game: GameType,
    agent: AgentType,
    tree_kwargs: dict[str, Any] = {},
    search_kwargs: dict[str, Any] = {},
) -> SelfPlayResult:
    """
    Play a game using a policy, and return the game states, action distributions, and final reward.
    """
    return asyncio.run(
        async_self_play(
            logger,
            state,
            game,
            agent,
            tree_kwargs,
            search_kwargs
        )
    )
