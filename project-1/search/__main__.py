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


def main():
    try:
        with open(sys.argv[1]) as file:
            data = json.load(file)
    except IndexError:
        print("usage: python3 -m search path/to/input.json", file=sys.stderr)
        sys.exit(1)

    game_board = Board(radius=5)
    game_board.populate_grid(data)
    game_board.print_grid()
    for piece in [*game_board.upper_pieces, *game_board.lower_pieces]:
        print('#', piece, piece.coord)
        print('#', piece.get_adjacent_hexes())


main()
