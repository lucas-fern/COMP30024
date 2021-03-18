from enum import Enum, auto


class Piece:
    def __init__(self, identifier, board):
        if identifier:
            self.team = Team.UPPER if identifier.isupper() else Team.LOWER
        else:
            self.team = Team.BLOCK

        self.symbol = Symbol(identifier.upper())  # Assigns a Symbol Enum based on the identifier string
        self.board = board

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
        return self.symbol.value if self.team == Team.UPPER else self.symbol.value.lower()


class Team(Enum):
    UPPER = auto()
    LOWER = auto()
    BLOCK = None


class Symbol(Enum):
    ROCK = 'R'
    PAPER = 'P'
    SCISSORS = 'S'
    BLOCK = ''
