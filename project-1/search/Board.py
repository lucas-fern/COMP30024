import numpy as np
import itertools
import sys
from search.util import print_board
from search.board_util import *


class Board:
    """A class to represent the hexagonal game board. The board is defined with a radius attribute so that we aren't
    constrained to playing on a radius 5 grid, though I think we will only ever play on a board this size.

    We want to work in centered coordinates wherever possible and only convert to array coordinates for accessing the
    pieces in the board grid."""
    def __init__(self, radius=5):
        self.radius = radius
        self.lower_pieces = {'r': [], 'p': [], 's': []}
        self.upper_pieces = {'R': [], 'P': [], 'S': []}
        self.blocked_coords = set()

        # Initialises a board of empty lists - the board will be filled according to
        # https://www.redblobgames.com/grids/hexagons/#map-storage
        # To visualise, imagine taking the diagram on p2 of spec-A.pdf and rotating the q axis to be horizontal
        # while keeping the hexes lined up with the dotted lines perpendicular to each axis.
        self.grid = np.empty((radius * 2 - 1, radius * 2 - 1), dtype=object)
        for row, col in np.ndindex(self.grid.shape):
            self.grid[row, col] = []

    def __getitem__(self, item):
        """Makes the board subscriptable, this means instead of having to access the grid to get the pieces in it with
            > board.grid[row, col]
        we can directly index the board object and it will return the same thing. eg.
            > board[row, col]
        and
            > board[row, col] == board.grid[row, col]
            True

        Can pass a coordinate into board[] as a tuple, eg. board[(1, 3)] that way we don't even need to deconstruct
        coordinates. This whole method is syntactic sugar but very nice :)
        """
        return self.grid[item[0], item[1]]

    def generate_token_moves(self):
        moves = []
        for identifier in self.upper_pieces:
            for from_tile in self.upper_pieces[identifier]:
                # Cant use sets since multiple identical pieces can exist on the same coordinate
                slide_moves = [(identifier, from_tile, Move.SLIDE, to_tile) for to_tile in
                               get_valid_slides(from_tile, self.radius, self.blocked_coords)]
                swing_moves = [(identifier, from_tile, Move.SWING, to_tile) for to_tile in
                               get_valid_swings(from_tile, identifier, self.grid, self.radius, self.blocked_coords)]
                moves.append((*slide_moves, *swing_moves))

        # Calculate all possible sets of moves for upper
        movement_options = list(itertools.product(*moves))

        print(movement_options)
        print("# Printing boards")

        board_set = []  # If we define Board.__hash__() then we can make this a set to remove duplicates.
        for move_set in movement_options:
            try:
                new_board = apply_moves(self, move_set)
                board_set.append(new_board)
            except IndexError:
                print("Board index out of range.", file=sys.stderr)
                # Probably shouldn't have any chances to exit() in final code, just continue anyway to get some marks
                sys.exit(1)
            except ValueError:
                print("Piece does not exist at coordinate.", file=sys.stderr)
                # Ditto above
                sys.exit(1)

        return board_set

    def add_piece(self, coordinate: tuple, identifier: str):
        """Adds a piece to the board grid, takes a centered coordinate and an identifier string for the piece."""
        if -self.radius < sum(coordinate) < self.radius:  # Valid positions on the board follow this rule!
            array_coord = centered_to_array_coord(coordinate, self.radius)
            # self passes in a reference to this game board to be stored by the Piece
            self.grid[array_coord].append(identifier)

            if identifier.islower():
                self.lower_pieces[identifier].append(coordinate)
            elif identifier.isupper():
                self.upper_pieces[identifier].append(coordinate)
            else:
                self.blocked_coords.add(coordinate)

    def populate_grid(self, initial_dict):
        """Fills the board grid from a dictionary of pieces and positions. The dictionary should be structured as
        if it was loaded from a JSON test case."""
        for piece in initial_dict['upper']:
            self.add_piece((piece[1], piece[2]), piece[0].upper())
        for piece in [*initial_dict['lower'], *initial_dict['block']]:  # Unpack and combine the two piece lists
            self.add_piece((piece[1], piece[2]), piece[0])

    def grid_to_dict(self) -> dict:
        """Converts the board grid to a dictionary of coordinate: piece(s) pairs. The dictionary is compatible with
        search.util.print_board()."""
        centered_dict = {}
        for row, col in np.ndindex(self.grid.shape):
            if self.grid[row, col]:
                string = ''.join(self.grid[row, col])
                centered_dict[array_to_centered_coord((row, col), self.radius)] = string

        return centered_dict

    def print_grid(self, compact=False) -> None:
        """Converts the board grid to a dictionary using grid_to_dict() and uses search.util.print_board() to print
        a visual representation of the board state."""
        print_dict = self.grid_to_dict()
        print_board(print_dict, compact=compact)
