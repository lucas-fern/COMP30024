from enum import Enum, auto


class Piece:
    ADJACENT_OFFSETS = ((+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1))

    def __init__(self, identifier, centered_coord, board):
        self.coord = centered_coord
        self.board = board

        if identifier:
            self.team = Team.UPPER if identifier.isupper() else Team.LOWER
            self.symbol = Symbol(identifier.upper())  # Assigns a Symbol Enum based on the identifier string
        else:
            self.team = Team.BLOCK
            self.symbol = Symbol.BLOCK

    def __repr__(self):
        """The string and command line representation of the Piece, what shows up if you enter
            > a = Piece(...)
            > a
        in the debug command line, or, since a __str__ method hasn't been defined, what comes up when we call
            > str(a)
        or
            > print(a)
        This allows us to just print the pieces directly in util.print_board().
        """
        return self.symbol.value.lower() if self.team == Team.LOWER else self.symbol.value

    def get_adjacent_hexes(self, centered_coord: tuple = None):
        """Gives a list of the valid hexes adjacent to the piece, takes a coordinate as an optional keyword argument
        and uses that coordinate instead of this piece's if provided. Might turn into a @staticmethod?

        This code is actually kind of sexy even if it is a bit complicated."""
        centered_coord = self.coord if not centered_coord else centered_coord
        # Zips together the coordinate with each valid adjacent offset (makes an iterator of tuple pairs)
        pairs = zip([centered_coord] * len(Piece.ADJACENT_OFFSETS), Piece.ADJACENT_OFFSETS)
        # Zips together the tuple pairs so that the row coordinates and the column coordinates are together
        zipped_pairs = [list(zip(*i)) for i in pairs]
        # Sums the row and column coordinates with the offsets and forms them into a coordinate tuple
        adjacent_hexes = [(sum(i[0]), sum(i[1])) for i in zipped_pairs]
        # Keeps only the coordinates which are on the board
        adjacent_hexes = [i for i in adjacent_hexes if -self.board.radius < sum(i) < self.board.radius]

        return adjacent_hexes


class Team(Enum):
    UPPER = auto()
    LOWER = auto()
    BLOCK = None


class Symbol(Enum):
    ROCK = 'R'
    PAPER = 'P'
    SCISSORS = 'S'
    BLOCK = 'B'
