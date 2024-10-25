from wayfinder.uct.uct_node import UCTNode

class TreeDiagnostics:
    def __init__(self, root: UCTNode):
        self.root = root

        # dictionary representation of subtree rooted at this node
        self.tree_dict = None

    async def get_tree_dict_repr(self) -> None:
        """
        Get the current tree structure.

        A dictionary representation of the tree. e.g. 
            {
                "A": {
                    "B": {
                        "D": {},
                        "E": {}
                    },
                    "C": {
                        "F": {},
                        "G": {}
                    }
                }
            }
        """
        tree = {
            "action_idx": self.root.action_idx, # the action that led to this node; -1 if root
            "number_visits": self.root.number_visits, # number of times this node has been visited
            "total_value": self.root.total_value, # total value of this node
            "initial_value": self.root.initial_value, # initial value of this node
            "impossible": self.root.impossible, # whether this node is impossible
            "children": {} # children of this node. the key is the action index, and the value is the representation of the child subtree
        }

        for action_idx, child in self.root.children.items():
            tree["children"][action_idx] = await child.get_tree_dict_repr()

        self.tree_dict = tree
    
    async def get_tree_labels(self) -> None:
        '''
        Generate a list of labels of all nodes in the tree.
        Each label is a string of the form "0.3.2.3. ..." as a record of which actions you take to get here from the root.

        Example: if the current tree dict repr is:
        {
            "children": {
            "0": {
                "children": {
                "0": {
                    "children": {}
                },
                "1": {
                    "children": {}
                }
                }
            },
            "1": {
                "children": {
                "0": {
                    "children": {}
                },
                "1": {
                    "children": {}
                }
                }
            }
            }
        }
            
        Then the labels would be:
        [
            "",
            "0.",
            "1.",
            "0.0.",
            "0.1.",
            "1.0.",
            "1.1."
        ]
        '''
        if not self.tree_dict:
            await self.get_tree_dict_repr()

        def dfs(node, label):
            node["label"] = label
            for action_idx, child in node["children"].items():
                dfs(child, label + str(action_idx) + ".")

        dfs(self.tree_dict, "")
    
    async def print_tree(self) -> str:
        '''
        Print the tree structure.

        Returns: a string containing 
        ---
        label

        info from tree dict

        children
        ---
        ...
        '''
        self.get_tree_labels()
        ret = ""
        
        def dfs(node):
            # node is a dictionary
            # print the label
            ret += f"---\n{node['label']}\n"
            for key, value in node.items():
                if key == "children":
                    continue
                ret += f"{key.ljust(20)}: {value}\n"
            # print out the labels of the children
            ret += "children labels:\n"
            for action_idx, child in node["children"].items():
                ret += f"{child['label']}\n"
            
            for action_idx, child in node["children"].items():
                dfs(child)