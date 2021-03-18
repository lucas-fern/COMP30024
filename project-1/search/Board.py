import numpy as np
from search.util import print_board
from search.Piece import *


class Board:
    """A class to represent the hexagonal game board. The board is defined with a radius attribute so that we aren't
    constrained to playing on a radius 5 grid, though I think we will only ever play on a board this size.

    We want to work in centered coordinates wherever possible and only convert to array coordinates for accessing the
    pieces in the board grid."""
    def __init__(self, radius=5):
        self.radius = radius
        self.lower_pieces = []
        self.upper_pieces = []
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

    def add_piece(self, coordinate: tuple, identifier: str):
        """Adds a piece to the board grid, takes a centered coordinate and an identifier string for the piece."""
        if -self.radius < sum(coordinate) < self.radius:  # Valid positions on the board follow this rule!
            array_coord = self.centered_to_array_coord(coordinate)
            # self passes in a reference to this game board to be stored by the Piece
            piece = Piece(identifier, coordinate, self)
            self.grid[array_coord[0], array_coord[1]].append(piece)

            if identifier.islower():
                self.lower_pieces.append(piece)
            elif identifier.isupper():
                self.upper_pieces.append(piece)
            else:
                self.blocked_coords |= {coordinate}  # | is the python set union operator (as well as logical or)

    def populate_grid(self, initial_dict):
        """Fills the board grid from a dictionary of pieces and positions. The dictionary should be structured as
        if it was loaded from a JSON test case."""
        for piece in initial_dict['upper']:
            self.add_piece((piece[1], piece[2]), piece[0].upper())
        for piece in [*initial_dict['lower'], *initial_dict['block']]:  # Unpack and combine the two piece lists
            self.add_piece((piece[1], piece[2]), piece[0])

    def centered_to_array_coord(self, coordinate: tuple) -> tuple:
        """Converts an axial coordinate with (0, 0) in the center of the grid to an axial coordinate with (0, 0) in the
        top left hex. Reverses the positive row direction so that increasing the row moves down the grid.

        Example:
            Centered Coordinates:
                _,-' `-._,-' `-._
                |       |       |
                |  1,-1 |  1, 0 |
            _,-' `-._,-' `-._,-' `-._
            |       |       |       |
            |  0,-1 |  0, 0 |  0, 1 |
             `-._,-' `-._,-' `-._,-'
                |       |       |
                | -1, 0 | -1, 1 |
                 `-._,-' `-._,-'

            Array Coordinates:
                _,-' `-._,-' `-._
                |       |       |
                |  0, 0 |  0, 1 |
            _,-' `-._,-' `-._,-' `-._
            |       |       |       |
            |  1, 0 |  1, 1 |  1, 2 |
             `-._,-' `-._,-' `-._,-'
                |       |       |
                |  2, 1 |  2, 2 |
                 `-._,-' `-._,-'

            Array Coordinates in Array:
            +--------+--------+--------+
            | (0, 0) | (0, 1) | None   |
            +--------+--------+--------+
            | (1, 0) | (1, 1) | (1, 2) |
            +--------+--------+--------+
            | None   | (2, 1) | (2, 2) |
            +--------+--------+--------+
                > Notice how the array coordinates correspond to a real array index that we can plug straight into
                > a numpy array to access that hex! nice

            """
        offset = (-(self.radius - 1), (self.radius - 1))
        translated = [sum(x) for x in zip(coordinate, offset)]

        return -translated[0], translated[1]  # Element wise sum of the coordinate and offset

    def array_to_centered_coord(self, coordinate: tuple) -> tuple:
        """Performs the inverse transformation to centered_to_array_coord(). Transforms coordinates from the array
        coordinate system to centered axial coordinates."""
        inverted = (-coordinate[0], coordinate[1])
        offset = ((self.radius - 1), -(self.radius - 1))

        return tuple(sum(x) for x in zip(inverted, offset))  # Element wise sum of the coordinate and offset

    def grid_to_dict(self) -> dict:
        """Converts the board grid to a dictionary of coordinate: piece(s) pairs. The dictionary is compatible with
        search.util.print_board()."""
        centered_dict = {}
        for row, col in np.ndindex(self.grid.shape):
            if self.grid[row, col]:
                centered_dict[self.array_to_centered_coord((row, col))] = self.grid[row, col]

        return centered_dict

    def print_grid(self, compact=False) -> None:
        """Converts the board grid to a dictionary using grid_to_dict() and uses search.util.print_board() to print
        a visual representation of the board state."""
        print_dict = self.grid_to_dict()
        print_board(print_dict, compact=compact)

