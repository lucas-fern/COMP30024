import numpy as np
import itertools
import random
import traceback
import sys
import copy
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
        self.heuristic_score = None
        self.children = None
        self.f = 0
        self.g = 0
        self.connecting_move_set = None
        self.parent = None

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

    def __lt__(self, other):
        """The less than comparison is used when sorting items in the priority queue"""
        return self.f < other.f

    def __eq__(self, other):
        return np.array_equal(self.grid, other.grid)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        string = ''
        for ix, iy in np.ndindex(self.grid.shape):
            string += str(sorted(self.grid[ix, iy]))
        return string

    def generate_children(self):
        """Generates all possible board states that can be reached from a single set of upper moves. Adds to children"""
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
        movement_options = itertools.product(*moves)

        self.children = []  # If we define Board.__hash__() then we can make this a set to remove duplicates.
        for move_set in movement_options:
            try:
                new_board = self.apply_moves(move_set)
                self.children.append(new_board)
            except IndexError:
                print("Board index out of range.", file=sys.stderr)
                # Probably shouldn't have any chances to exit() in final code, just continue anyway to get some marks
                sys.exit(1)
            except ValueError:
                print("Piece does not exist at coordinate.", file=sys.stderr)
                print(traceback.format_exc())
                # Ditto above
                sys.exit(1)

    def apply_moves(self, move_set) -> 'Board':
        """Applies a move set from the output of Board.generate_token_moves(). Moves pieces regardless of whether moves
        are valid. Returns a new board object with the pieces moved."""
        new_board = self.spawn_offspring()
        for symbol, from_coord, move_type, to_coord in move_set:
            array_coord_from = centered_to_array_coord(from_coord, new_board.radius)
            array_coord_to = centered_to_array_coord(to_coord, new_board.radius)

            new_board.grid[array_coord_from].remove(symbol)
            new_board.grid[array_coord_to].append(symbol)

            new_board.upper_pieces[symbol].remove(from_coord)
            new_board.upper_pieces[symbol].append(to_coord)

        new_board.battle()  # Lets remove the dead pieces here so we aren't passing around half-completed moves
        new_board.set_heuristic()
        new_board.connecting_move_set = move_set
        new_board.parent = self
        return new_board

    def spawn_offspring(self) -> 'Board':
        """Creates a child board without any children of its own and no heuristic score."""
        offspring = Board(self.radius)
        offspring.upper_pieces = copy.deepcopy(self.upper_pieces)
        offspring.lower_pieces = copy.deepcopy(self.lower_pieces)
        offspring.blocked_coords = copy.deepcopy(self.blocked_coords)
        offspring.grid = copy.deepcopy(self.grid)

        offspring.heuristic_score = None
        offspring.children = []

        return offspring

    def add_piece(self, coordinate: tuple, identifier: str):
        """Adds a piece to the board grid, takes a centered coordinate and an identifier string for the piece."""
        if valid_centered_hex(coordinate, self.radius):
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
        self.set_heuristic()

    def grid_to_dict(self) -> dict:
        """Converts the board grid to a dictionary of coordinate: piece(s) pairs. The dictionary is compatible with
        search.util.print_board()."""
        centered_dict = {}
        for row, col in np.ndindex(self.grid.shape):
            if self.grid[row, col]:
                pieces = self.grid[row, col]
                string = ''.join([piece if piece else 'Block' for piece in pieces])
                centered_dict[array_to_centered_coord((row, col), self.radius)] = string

        return centered_dict

    def print_grid(self, compact=False) -> None:
        """Converts the board grid to a dictionary using grid_to_dict() and uses search.util.print_board() to print
        a visual representation of the board state."""
        print_dict = self.grid_to_dict()
        print_board(print_dict, compact=compact)

    def battle(self):
        """Kills any overlapping pieces in the board grid according to the rules of RoPaSci360."""
        # Could be done much more cleanly but works for now, will worry about optimisation later
        for idx, pieces in np.ndenumerate(self.grid):
            if len(pieces) > 1:
                # print("battle on", idx, [c.lower() for c in x])
                # Check for existence of all 3 pieces on hex
                if all(symbols in [piece.lower() for piece in pieces] for symbols in ['r', 'p', 's']):
                    for piece in pieces:
                        # I think we need to cover the case where multiple of these pieces exist, so:
                        coord = array_to_centered_coord(idx, self.radius)
                        while coord in self.upper_pieces[piece.upper()]:
                            self.upper_pieces[piece.upper()].remove(coord)
                        while coord in self.lower_pieces[piece.lower()]:
                            self.lower_pieces[piece.lower()].remove(coord)
                    # All pieces are killed on the hex
                    self.grid[idx] = []

                    continue  # Skip to the next loop iteration if all the pieces in the cell are dead :)

                # Loop over all the killing combinations and kill the relevant tokens if they exist.
                for killer, killed in (('r', 's'), ('s', 'p'), ('p', 'r')):
                    if killer in [piece.lower() for piece in pieces]:
                        coord = array_to_centered_coord(idx, self.radius)
                        while coord in self.lower_pieces[killed]:
                            self.lower_pieces[killed].remove(coord)
                            self.grid[idx].remove(killed)
                        while coord in self.upper_pieces[killed.upper()]:
                            self.upper_pieces[killed.upper()].remove(coord)
                            self.grid[idx].remove(killed.upper())

    def is_game_over(self):
        """Checks if the game is over by seeing if either team has lost all of their pieces."""
        if not [a for a in self.lower_pieces.values() if a] or not \
               [a for a in self.upper_pieces.values() if a]:
            return True
        return False

    def set_heuristic(self):
        """Sets the heuristic as the sum of the BFS distances between all lower piece and their
        closest opponent killer."""
        self.heuristic_score = 0
        for symbol, coordinates in self.lower_pieces.items():
            if not coordinates:
                continue

            killer_symbol = KILL_RELATIONS[symbol]
            for lower_coord in coordinates:
                closest_killer_dist = float('inf')
                killer_coords = self.upper_pieces[killer_symbol.upper()]
                for killer_coord in killer_coords:
                    closest_killer_dist = min(closest_killer_dist, self.bfs_distance(lower_coord, killer_coord))
                # it is important to break ties between two equidistant targets, so random noise is added.
                self.heuristic_score += closest_killer_dist

        self.heuristic_score += random.uniform(0, 0.001)

    def bfs_distance(self, a, b):
        """Uses BFS to return the distance between two coordinates around any obstacles. Algorithm adapted from
        https://www.redblobgames.com/pathfinding/a-star/introduction.html"""

        frontier = [a]
        came_from = dict()
        came_from[a] = None
        current = None

        while frontier:
            current = frontier.pop(0)

            if current == b:
                break

            for next_hex in get_valid_slides(current, self.radius, self.blocked_coords):
                if next_hex not in came_from:
                    frontier.append(next_hex)
                    came_from[next_hex] = current

        distance = 0
        while current != a:
            current = came_from[current]
            distance += 1

        return distance
