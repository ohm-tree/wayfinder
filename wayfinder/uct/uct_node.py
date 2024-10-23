"""
uct_node.py

This module contains the UCTNode class, which represents a node in the UCT search tree.

Nodes can be in a couple of states.
"""

import asyncio
from calendar import c
from typing import Any, Generic, Hashable, Iterator, Optional, TypeVar

import numpy as np

from wayfinder.games import *

MoveType = TypeVar('MoveType', bound=Hashable)
StateType = TypeVar('State', bound=State)
GameType = TypeVar('GameType', bound=Game[MoveType, StateType])
AgentType = TypeVar('AgentType', bound=Agent[GameType, StateType, MoveType])


class UCTNode(Generic[GameType, StateType, AgentType]):
    def __init__(self,
                 agent: AgentType,
                 game: GameType,
                 state: StateType,
                 action_idx: int,
                 parent: 'UCTNode[GameType, StateType, AgentType]' = None,
                 init_type: str = "zero",
                 c: float = 1.0,
                 noise: bool = True,
                 ):
        """
        Initialize a new UCTNode.
        """
        # Reference to the agent, game, and state.
        self.agent: AgentType = agent
        self.game: GameType = game
        self.state: StateType = state

        # Action to enter the node, -1 if root
        self.action_idx: int = action_idx

        # Parent of the node, None if root
        self.parent: Optional[UCTNode] = parent

        # The priors and values are obtained from a neural network every time you expand a node
        # The priors, total values, and number visits will be 0 on all illegal actions
        self.child_priors: np.ndarray = np.zeros((0,), dtype=np.float64)
        self.child_total_value: np.ndarray = np.zeros((0,), dtype=np.float64)
        self.child_number_visits: np.ndarray = np.zeros((0,), dtype=np.int32)

        self.impossible = False

        # This is a snapshot of the original neural network value of the node.
        self.initial_value = None

        # This is a dictionary of action -> UCTNode. Only legal actions are keys
        self.children: dict[int, UCTNode] = {}

        # Used iff you are the root.
        if self.parent is None:
            self.root_total_value: float = 0
            self.root_number_visits: int = 0

        self.init_type = init_type
        self.c = c
        self.noise = noise

        # Invariant: if you are value_lock-ed, then:
        #  - We are currently requesting moves from the agent at this node.
        #  - This is to prevent multiple redundant requests.
        #  - However, other crawlers should still be able to traverse through this node to other children.
        self.value_lock = asyncio.Lock()
        self.move_lock = asyncio.Lock()

    async def root(self) -> None:
        """
        Set the parent of the node to None.
        """
        if self.parent is not None:
            # I was not originally the root, so actually I have some data to pass down.
            self.root_total_value = self.total_value
            self.root_number_visits = self.number_visits
        else:
            # I am the root, and I never had a parent, so I should initialize these values.
            self.root_total_value = 0
            self.root_number_visits = 0
        self.parent = None
        self.action_idx = -1

        await self.game.make_root(self.state)

    @property
    def valued(self) -> bool:
        """
        Returns whether self.initial_value has been set
        (through either termination or a value estimate in self.backup).
        """
        return self.initial_value is not None

    @property
    def expand_initial_value(self) -> float:
        """        
        Recommended by minigo which cites Leela.
        See https://github.com/tensorflow/minigo/blob/master/cc/mcts_tree.cc line 448.
        See https://www.reddit.com/r/cbaduk/comments/8j5x3w/first_play_urgency_fpu_parameter_in_alpha_zero/
        """

        value = 0

        if self.init_type == "zero":
            value = 0
        if self.init_type == "value":
            value = self.initial_value
        if self.init_type == "offset":
            value = self.initial_value - 0.1

        return value

    @property
    def expanded(self) -> bool:
        """
        Returns whether the node has been expanded.
        """
        return len(self.children) > 0

    @property
    def number_visits(self) -> int:
        if self.parent is None:
            return self.root_number_visits
        return self.parent.child_number_visits[self.action_idx]

    @number_visits.setter
    def number_visits(self, value) -> None:
        if self.parent is None:
            self.root_number_visits = value
        else:
            self.parent.child_number_visits[self.action_idx] = value

    @property
    def total_value(self):
        if self.parent is None:
            return self.root_total_value
        return self.parent.child_total_value[self.action_idx]

    @total_value.setter
    def total_value(self, value) -> None:
        if self.parent is None:
            self.root_total_value = value
        else:
            self.parent.child_total_value[self.action_idx] = value

    async def terminal(self) -> bool:
        """
        Returns whether the current node is terminal.

        This is different from the game._terminal function,
        because a node is also considered terminal if the agent is unable
        to find a valid move.
        """
        return self.impossible or await self.game._terminal(self.state)

    def reward(self) -> bool:
        """
        Returns whether the current node is terminal.

        This is different from the game._terminal function,
        because a node is also considered terminal if the agent is unable
        to find a valid move.
        """
        if self.impossible:
            return self.game.death_value
        else:
            return self.game._reward(self.state)

    def child_Q(self) -> np.ndarray:
        """
        The value estimate for each child, based on the average value of all visits.
        """
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self) -> np.ndarray:
        """
        The uncertainty for each child, based on the UCT formula (think UCB).
        """
        return np.sqrt(1 + np.sum(self.child_number_visits)) * (self.child_priors / (1 + self.child_number_visits))

    async def request_moves(self) -> None:
        """
        Determine whether we should ask the Agent for more legal moves.

        If so, request the moves and set the new policy. No need to change
        and value estimates here.

        As a baseline, the following implementation ensures that there are
        approximately sqrt(N) legal moves when the
        current node has been visited N times.
        """

        assert self.valued, "Cannot request moves from an unvalued node."

        if self.expanded and self.move_lock.locked():
            """
            If we are expanded, then there exists some child. The node is unavailable,
            meaning somebody else is making a request for moves. We shouldn't make a request
            ourselves, but there's no need to wait for the request to finish; we can just
            branch to the available child.
            """
            return

        min_request, max_request = await self.agent.amount_to_request(
            state=self.state,
            current_num_children=len(self.children),
            num_visits=self.number_visits,
            child_num_visits=self.child_number_visits,
            child_priors=self.child_priors,
            child_total_value=self.child_total_value,
            child_Q=self.child_Q(),
            child_U=self.child_U(),
            c=self.c,
            expand_initial_value=self.expand_initial_value,
        )

        # print(f"Requesting moves from {min_request} to {max_request}")
        """
        Temporarily flag the node as unavailable to prevent multiple requests for new moves.
        Critical section is only activated conditioned on actually making this request.
        Two cases: if not self.expanded, then we need to wait no matter what.
        if we're expanded but it was unlocked, we will obtain the lock immediately and request moves.
        """
        if len(self.children) < max_request:
            # print("Requesting moves")
            async with self.move_lock:
                # print("Inside critical section")
                success = await self.agent.require_new_move(self.state, min_request, max_request)
                if success:
                    # print("Success")
                    await self.expand()
                else:
                    # print("Failure")
                    # In this edge case, the agent is unable to find any legal moves.
                    # We should mark this node as terminal.
                    self.impossible = True

    def best_child(self) -> 'UCTNode[GameType, StateType, AgentType]':
        """
        Compute the best child.
        """
        scores = self.child_Q() + self.c * self.child_U()

        # scores[~self.action_mask] = -np.inf

        return self.children[np.argmax(scores)]

    async def select_leaf(self, virtual_loss=True) -> 'UCTNode[GameType, StateType, AgentType]':
        """
        Deterministically select the next leaf to expand based on the best path.

        Parameters:
        ----------
        virtual_loss: bool
            Whether to add a virtual loss to the selected node.
        """
        current = self

        assert not await self.terminal(), "Cannot select a leaf from a terminal node."

        # iterate until either you reach an un-expanded node or a terminal state
        while (not await current.terminal()):
            # If this takes a long time, it means that the current node is being valued,
            # and we should wait for it to finish. We should not attempt to value it as well.
            await current.value_lock.acquire()
            if current.valued:
                # We don't need the lock anymore.
                current.value_lock.release()
            else:
                # The node has never been valued!
                # Do not release the lock until we value it.
                break

            await current.request_moves()

            # edge case: if the node is impossible, then we should return it.
            if current.impossible:
                # release the lock, and return the current node.
                current.value_lock.release()
                break

            # Add a virtual loss.
            if virtual_loss:
                current.number_visits += 1
                current.total_value += self.game.death_value

            assert len(current.children) > 0, "Node has no children."
            current = current.best_child()

        # Add a virtual loss.
        if virtual_loss:
            current.number_visits += 1
            current.total_value += self.game.death_value

        return current

    async def expand(self) -> None:
        """
        Expand a non-terminal node using the child_priors from the neural network.

        Queries self.agent for the policy estimate of all of the children.
        """
        # print("Inside expand")

        assert self.move_lock, "expand() called without critical section protection."

        assert not await self.terminal(), "Cannot expand a terminal node."

        assert self.initial_value is not None, "Node has not been backed up, so I don't know the initial NN value."
        assert self.valued, "Node has not been valued."

        value = self.expand_initial_value

        old_length = len(self.children)
        new_length = await self.agent.len_active_moves(self.state)
        assert new_length >= old_length, "New length is less than old length."
        assert new_length > 0, "New length is 0."

        for action_idx in range(old_length, new_length):
            move = await self.agent.get_active_move(self.state, action_idx)
            # if self.action_mask[action_idx]:
            await self.add_child(action_idx,
                                 await self.agent._policy(self.state, move),
                                 value)

    async def add_child(self, action_idx, prior, value) -> None:
        """
        Add a child with a given action and prior probability.

        The value_estimates are updated together in expand().

        Parameters:
        ----------
        action_idx: int
            The action index of the child.
        prior: float
            The prior probability of the child.
        value: float
            The value estimate of the child.
        """
        assert not action_idx in self.children, f"Child with action {action_idx} already exists."

        action = await self.agent.get_active_move(self.state, action_idx)
        next_state = await self.game._next_state(self.state, action)

        # append to these numpy arrays

        # This is all done over here, so that everything happens atomically;
        # else in best_child, we might end up with a P or U value that doesn't
        # correspond to an element of self.children.
        if self.child_priors is None:
            self.child_priors = np.array([prior])
            self.child_total_value = np.array([value])
            self.child_number_visits = np.array([0])
        else:
            self.child_priors = np.append(self.child_priors, [prior])
            self.child_total_value = np.append(self.child_total_value, [value])
            self.child_number_visits = np.append(self.child_number_visits, [0])

        self.children[action_idx] = UCTNode(
            agent=self.agent,
            game=self.game,
            state=next_state,
            action_idx=action_idx,
            parent=self,
            init_type=self.init_type,
            c=self.c,
            noise=self.noise,
        )

    async def backup(self, estimate) -> None:
        """
        Propagate the estimate of the current node,
        back up along the path to the root.

        Parameters:
        ----------
        estimate: float
            The value estimate of the current node.
        """
        self.initial_value = estimate

        if not await self.terminal():
            # If i'm terminal, then I should have released the value lock a while ago.
            # hey, i have a value estimate now, so i can release the value lock.
            self.value_lock.release()
        # This should execute immediately after the preceding if statement.
        assert not self.value_lock.locked(), "Value lock is still locked."
        current = self
        while current.parent is not None:
            # Do not increment the number of visits here, because it is already done in select_leaf.
            # Extra +1 to the estimate to offset the virtual loss.
            current.total_value += estimate - self.game.death_value
            current = current.parent


async def dirichlet_noise(action_mask, alpha) -> np.ndarray:
    """
    Returns a dirichlet noise distribution on the support of an action mask.

    Parameters:
    ----------
    # action_mask: np.ndarray
        A boolean array of legal actions.

    alpha: float
        The dirichlet noise parameter.
    """

    places = action_mask > 0
    noise = np.random.dirichlet(alpha * np.ones(np.sum(places)))

    noise_distribution = np.zeros_like(action_mask, dtype=np.float64)
    noise_distribution[places] = noise

    return noise_distribution
