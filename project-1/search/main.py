"""
COMP30024 Artificial Intelligence, Semester 1, 2021
Project Part A: Searching
"""

import sys
import json
import time
import heapq
from search.util import print_slide, print_swing
from search.Board import Board
from search.board_util import *


def main(data=None):
    start_time = time.time()
    if data is None:
        # Load the JSON file with the board
        try:
            with open(sys.argv[1]) as file:
                data = json.load(file)
        except IndexError:
            print("usage: python3 -m search path/to/input.json", file=sys.stderr)
            sys.exit(1)

    # Initialise the board with the data
    game_board = Board(radius=5)
    game_board.populate_grid(data)
    game_board.print_grid()

    # A* Algorithm
    open_set = set()
    open_heap = []
    closed_set = set()
    open_set.add(game_board)
    heapq.heappush(open_heap, game_board)
    current_node = None

    while len(open_set) > 0:
        current_node = heapq.heappop(open_heap)
        open_set.remove(current_node)
        closed_set.add(current_node)

        # print(len(open_set), len(open_heap), len(closed_set))

        # current_node.print_grid()
        if current_node.is_game_over():
            break

        # Add children of the current node to the list of nodes to be explored
        current_node.generate_children()
        for child in current_node.children:
            # Generate g(n) and f(n) = g(n) + heuristic(n)
            child.g = current_node.g + 1
            child.f = child.g + child.heuristic_score

            if (child in closed_set) or (child in open_set):
                continue

            open_set.add(child)
            heapq.heappush(open_heap, child)

        # Time out after 30s for testing
        if time.time() - start_time > 30:
            print("Timed Out", file=sys.stderr)
            return False

    # Store the final board
    terminal_board = current_node

    # Work back from the terminal node and trace back the moves taken to get to this game state
    path = []
    while current_node is not None:
        path.append(current_node.connecting_move_set)
        current_node = current_node.parent

    # Print the path, in reverse order, starting with the second last element (because the root node is an orphan)
    path = path[-2::-1]
    for turn, moves in enumerate(path, start=1):
        for symbol, from_hex, move_type, to_hex in moves:
            if move_type == Move.SLIDE:
                print_slide(turn, from_hex[0], from_hex[1], to_hex[0], to_hex[1])
            if move_type == Move.SWING:
                print_swing(turn, from_hex[0], from_hex[1], to_hex[0], to_hex[1])

    terminal_board.print_grid()

    return True

