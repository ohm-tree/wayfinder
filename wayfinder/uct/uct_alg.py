"""
uct_alg.py

This module contains functions for running the UCT algorithm.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

import asyncio
import logging
import time
from typing import Any, Generic, Hashable, Iterator, Optional, TypedDict, TypeVar

import numpy as np

from wayfinder.games import *
from wayfinder.uct.uct_node import UCTNode

MoveType = TypeVar('MoveType', bound=Hashable)
StateType = TypeVar('State', bound=State)
GameType = TypeVar('GameType', bound=Game[MoveType, StateType])
AgentType = TypeVar('AgentType', bound=Agent[GameType, StateType, MoveType])


class crawler_shared_state:
    def __init__(self):
        self.victorious_death = False
        self.winning_node = None
        self.iters = 0


async def crawler(
    logger: logging.Logger,
    root: UCTNode[GameType, StateType, AgentType],
    shared_state: crawler_shared_state,
    num_iters: int = 100,
    min_log_delta: float = 10.0,  # 10 seconds between logging to not spam the logs
) -> None:
    """
    Crawl the UCT tree from the given root node.
    """
    last_log_time = time.time()

    # while not victorious_death and iters < num_iters:
    while shared_state.victorious_death is False and shared_state.iters < num_iters:
        shared_state.iters += 1

        # greedily select leaf with given exploration parameter
        leaf = await root.select_leaf()

        if (await leaf.terminal()):
            # Immediately backup the value estimate along the path to the root
            leaf.backup(await leaf.reward())

            if (await leaf.game._reward(leaf.state)):
                shared_state.victorious_death = True
                shared_state.winning_node = leaf

        elif not leaf.valued:
            assert leaf.value_lock.locked()

            # Request the value of the leaf from the agent
            await leaf.backup(await leaf.agent._value(leaf.state))
        else:
            raise ValueError(
                "Leaf should either be terminal or unvalued."
            )
        if time.time() - last_log_time > min_log_delta:
            last_log_time = time.time()


class UCTSearchResult(Generic[GameType, StateType, AgentType]):
    active_moves: list[MoveType]
    visit_distribution: np.ndarray
    best_Q: float
    winning_node: Optional[UCTNode[GameType, StateType, AgentType]]


async def async_uct_search(
    logger: logging.Logger,
    root: UCTNode[GameType, StateType, AgentType],
    num_iters: int = 100,
) -> UCTSearchResult[GameType, StateType, AgentType]:
    """
    Perform num_iters iterations of the UCT algorithm from the given game state
    using the exploration parameter c. Return the distribution of visits to each direct child.

    Requires that game_state is a non-terminal state.
    """

    shared_state = crawler_shared_state()

    # start up 10 crawlers
    # TODO: make the number of crawlers a search parameter
    crawlers = []
    for _ in range(10):
        crawlers.append(crawler(
            logger=logger,
            root=root,
            shared_state=shared_state,
            num_iters=num_iters,
        ))

    await asyncio.gather(*crawlers)

    return {
        "active_moves": root.agent.active_moves(root.state),
        "visit_distribution": root.child_number_visits / np.sum(root.child_number_visits),
        "best_Q": root.child_Q()[root.child_number_visits.argmax()],
        "winning_node": shared_state.winning_node
    }


def uct_search(
    logger: logging.Logger,
    root: UCTNode[GameType, StateType, AgentType],
    num_iters: int,
    train: bool = True,
) -> UCTSearchResult[GameType, StateType, AgentType]:
    """
    Perform num_iters iterations of the UCT algorithm from the given game state
    using the exploration parameter c. Return the distribution of visits to each direct child.

    Requires that game_state is a non-terminal state.
    """
    return asyncio.run(
        async_uct_search(
            logger,
            root,
            num_iters,
            train
        )
    )
