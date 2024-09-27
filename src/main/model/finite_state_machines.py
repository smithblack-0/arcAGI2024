mode_select = GraphNode()

class Variable:
    """
    Can contain a number
    """
    def set(self, value: int):
        self.value = value
    def __init__(self):
        self.value = None

class DirectedGraphNode:




def bind_text_mode(mode_select, vocabulary_size):
    transition = mode_select.new_link(trigger="text")
    transition.insert_context(mode="text")
    transition.emit_context()

    text_shape_select = transition.node()

    number = Variable()
    last_position_node = Node(mode="text", submode=0)
    transition = last_position_node.new_link(node=mode_select,trigger=number)
    transition.insert_context(data=number)
    transition.emit_context()
    transition.reset_context()

    for i in range(1, vocabulary_size):


        node = Node(mode="text")

        # Define the transition to the next position
        number = Variable()
        transition = node.new_link(trigger=number, node=last_position_node)
        transition.insert_context(data=number)
        transition.emit_context()

        # Define the transition into this from the shape select
        number = Variable()
        transition = text_shape_select.new_link(trigger=i, node=node)
        transition.insert_context(shape_x=i)
        transition.emit_context()




