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

# {**game_board.upper_pieces, **game_board.lower_pieces} is a merge of the two dictionaries, in python 3.9 we can use
# game_board.upper_pieces | game_board.lower_pieces, but I don't know if that's what you're using, and I doubt its
# what the assessment server will be using.
all_pieces = {**game_board.upper_pieces, **game_board.lower_pieces}

for identifier in all_pieces:
    coordinates = all_pieces[identifier]
    for coordinate in coordinates:
        print('#', identifier, coordinate)
        print('# adjacent hexes: ', get_adjacent_hexes(coordinate, game_board.radius))
        print('# valid slide moves: ', get_valid_slides(coordinate, game_board.radius, game_board.blocked_coords))
        print('# adjacent friendly hexes: ', get_adjacent_friendlies(coordinate, identifier, game_board.grid,
                                                                     game_board.radius))
        print('# valid swing moves: ', get_valid_swings(coordinate, identifier, game_board.grid,
                                                        game_board.radius, game_board.blocked_coords))

# A* algo

open_list = []
closed_list = []
open_list.append(game_board)

while len(open_list) > 0:
    current_node = open_list[0]
    current_index = 0
    # expand, preffering nodes with lowver combined heuristic scores
    for index, item in enumerate(open_list):
        if item.f < current_node.f:
            current_node = item
            current_index = index
    open_list.pop(current_index)
    closed_list.append(current_node)

    if current_node.is_game_over():
        print("Woo found winning set of moves")
        break

    current_node.generate_children()
    for child in current_node.children:
        for closed_child in closed_list:
            if child == closed_child:
                continue
        child.g = current_node.g + 1
        child.set_heuristic
        child.f = child.g + child.heuristic_score

        for open_node in open_list:
            if child == open_node and child.g > open_node.g:
                continue

        open_list.append(child)

path = []
current = current_node
while current is not None:
    path.append(current.connecting_move_set)
    current.print_grid(compact=True)
    current = current.parent
print(path[::-1]) # Return reversed path

current_node.print_grid()

