from typing import Callable, Generic, Hashable, Optional, TypeVar

from wayfinder.games.agent import Agent
from wayfinder.games.game import Game, State
from wayfinder.uct.uct_node import UCTNode

MoveType = TypeVar('MoveType', bound=Hashable)
StateType = TypeVar('State', bound=State)
GameType = TypeVar('GameType', bound=Game[MoveType, StateType])
AgentType = TypeVar('AgentType', bound=Agent[GameType, StateType, MoveType])


class TreeDiagnostics(Generic[MoveType, StateType, GameType, AgentType]):
    def __init__(
            self,
            game: GameType,
            game_state_to_string: Optional[Callable[[
                GameType, StateType], str]] = None
    ):
        self.game = game
        self.game_state_to_string = game_state_to_string

    def _node_to_string(
        self,
        node: UCTNode[MoveType, StateType, GameType, AgentType],
        depth: int,
        index_path
    ):
        """
        Convert a UCTNode to a string representation for debugging.

        Parameters
        ----------
        node : UCTNode
            The node to convert to a string.
        depth : int
            The depth of the node in the tree.
        index_path : tuple
            The path of indices leading to this node.
        """
        res = ""
        res += "-" * 80 + "\n"
        res += "-" * 80 + "\n"
        data = {
            "depth": depth,
            "index_path": index_path,
            "number_visits": node.number_visits,
            "hash": node.state.__hash__(),
            "total_value": node.total_value,
            "child priors": node.child_priors,
            "child number visits": node.child_number_visits,
            "child total value": node.child_total_value
        }
        for k, v in data.items():
            res += f"{k.ljust(20)} | {str(v)}\n"
        res += "-" * 80 + "\n"

        if self.game_state_to_string is not None:
            res += self.game_state_to_string(self.game, node.state) + "\n"
        else:
            res += str(node.state) + "\n"
        return res

    def tree_to_string(
        self,
        node: UCTNode[MoveType, StateType, GameType, AgentType],
        depth: int = 0,
        index_path=()
    ):
        """
        Convert the entire tree rooted at the given node to a string representation.

        Parameters
        ----------
        node : UCTNode
            The root node of the tree to convert to a string.
        depth : int
            The current depth in the tree.
        index_path : tuple
            The path of indices leading to this node.
        """
        res = self._node_to_string(node, depth, index_path)

        for idx, item in enumerate(node.children.items()):
            move, child_node = item
            res += self.tree_to_string(child_node, depth + 1,
                                       index_path + (idx,))
        return res
