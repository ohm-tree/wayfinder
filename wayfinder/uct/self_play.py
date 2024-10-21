"""
self_play.py

This module contains the self-play function.
This returns the game states and action distributions, as well as the final result.

"""

import asyncio
import logging
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
    tree_kwargs: Optional[dict[str, Any]] = None,
    search_kwargs: Optional[dict[str, Any]] = None,
) -> SelfPlayResult:
    """
    Play a game using a policy, and return the game states, action distributions, and final reward.
    """

    states: list[Any] = []
    distributions: list[np.ndarray] = []

    agent = Agent(game)

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

    while not (await game._terminal(root.state)):
        logger.info("Move: " + str(move_count))
        move_count += 1
        logger.info(root.state.__str__())
        """
        TODO: Fast Playouts would be implemented here.
        """
        winning_node: UCTNode
        distribution, _, winning_node = await async_uct_search(
            logger=logger,
            root=root,
            **search_kwargs
        )

        # TODO: more configurations possible for uct_search, not used right now.

        """
        TODO: In MCTS algorithms, people sometimes change up the temperature right here,
        to sharpen the training distribution. This is something we could try.
        """

        if winning_node is not None:
            root = winning_node
            break
        distributions.append(distribution)
        logger.info(f"Action distribution: {distribution}")

        action = np.random.choice(len(distribution), p=distribution)
        root = root.children[action]
        # set root parent to None so that it knows it is the root.
        root.root()
        states.append(root.state)

    logger.info("Move: " + str(move_count))
    logger.info(root.state.__str__())

    # The reward for all states in the tree is the reward of the final state.

    final_reward = await game._reward(root.state)
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
        "rewards": rewards
    }


def self_play(
    logger: logging.Logger,
    state: StateType,
    game: GameType,
    tree_kwargs: Optional[dict[str, Any]] = None,
    search_kwargs: Optional[dict[str, Any]] = None,
) -> SelfPlayResult:
    """
    Play a game using a policy, and return the game states, action distributions, and final reward.
    """
    return asyncio.run(
        async_self_play(
            logger,
            state,
            game,
            tree_kwargs,
            search_kwargs
        )
    )
