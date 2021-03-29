"""
COMP30024 Artificial Intelligence, Semester 1, 2021
Project Part A: Searching

This script contains the entry point to the program (the code in
`__main__.py` calls `main()`). Your solution starts here!
"""

import sys
import json

# If you want to separate your code into separate files, put them
# inside the `search` directory (like this one and `util.py`) and
# then import from them like this:
from search.util import print_board, print_slide, print_swing
from search.Board import Board
from search.board_util import *


try:
    with open(sys.argv[1]) as file:
        data = json.load(file)
except IndexError:
    print("usage: python3 -m search path/to/input.json", file=sys.stderr)
    sys.exit(1)

game_board = Board(radius=5)
game_board.populate_grid(data)
game_board.print_grid()

# A* Algorithm

open_list = []
closed_list = []
open_list.append(game_board)
current_node = None

while len(open_list) > 0:
    current_node = open_list[0]
    # Explore the open board with the lowest heuristic score
    for open_board in open_list:
        if open_board.f < current_node.f:
            current_node = open_board
    open_list.remove(current_node)
    closed_list.append(current_node)

    if current_node.is_game_over():
        print("# Woo found winning set of moves!")
        break

    # Add children of the current node to the list of nodes to be explored
    current_node.generate_children()
    for child in current_node.children:
        is_closed = False
        for closed_child in closed_list:
            # If the child has already been closed, set a flag to skip the outer loop, then break the inner loop
            if child == closed_child:
                is_closed = True
                break
        if is_closed:
            continue

        # The child isn't closed yet, so generate g(n) and f(n) = g(n) + heuristic(n)
        child.g = current_node.g + 1
        child.f = child.g + child.heuristic_score

        # If the child is already in the open set, and that version has a lower g(n) cost, don't add this one.
        superior_is_open = False
        for open_node in open_list:
            if child == open_node:
                if child.g > open_node.g:
                    superior_is_open = True
                    break
                else:
                    open_list.remove(open_node)
        if superior_is_open:
            continue

        open_list.append(child)

path = []
while current_node is not None:
    path.append(current_node.connecting_move_set)
    current_node.print_grid()
    current = current_node.parent
# Print the path, in reverse order, starting with the second last element (because the root node is an orphan)
print(path[-2::-1])

current_node.print_grid()

