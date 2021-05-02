import numpy as np
import itertools, time
import random
import traceback
import sys
import copy
from team_name.search.util import print_board
from team_name.search.board_util import *


class Board:
    """A class to represent the hexagonal game board. The board is defined with a radius attribute so that we aren't
    constrained to playing on a radius 5 grid, though I think we will only ever play on a board this size.

    We want to work in centered coordinates wherever possible and only convert to array coordinates for accessing the
    pieces in the board grid."""

    def __init__(self, radius=5, parent = None, turn = 'upper', thrown = {'upper': 0, 'lower': 0}, last_action = None, n_turns = 1):
        self.radius = radius
        self.parent = parent
        self.turn = turn
        self.thrown = thrown
        self.last_action = last_action
        self.winner = None
        self.terminal = False
        self.n_turns = n_turns

        # Initialises a board of empty lists - the board will be filled according to
        # https://www.redblobgames.com/grids/hexagons/#map-storage
        # To visualise, imagine taking the diagram on p2 of spec-A.pdf and rotating the q axis to be horizontal
        # while keeping the hexes lined up with the dotted lines perpendicular to each axis.
        self.grid = np.empty((radius * 2 - 1, radius * 2 - 1), dtype=object)
        for row, col in np.ndindex(self.grid.shape):
            self.grid[row, col] = []
        self.lower_pieces = {'r': [], 'p': [], 's': []}
        self.upper_pieces = {'R': [], 'P': [], 'S': []}

    def find_children(self):
        if self.terminal:  # If the game is finished then no moves can be made
            return set()
        moves = self.generate_moves()
        children = []
        for move in moves:
            child = Board(radius=self.radius, parent=self, turn=self.turn, last_action=move, n_turns=self.n_turns + 1)
            child.upper_pieces, child.lower_pieces, child.grid, child.thrown = copy.deepcopy(self.upper_pieces), copy.deepcopy(self.lower_pieces),\
                                                                 copy.deepcopy(self.grid), copy.deepcopy(self.thrown)
            child.apply_move(move)
            if self.turn == 'upper':
                child.turn = 'lower'
            else:
                child.turn = 'upper'
            child.set_winner_terminal()
            if move[0] == "THROW":
                child.thrown[self.turn] += 1
            children.append(child)
        return children

    def print_grid(self, compact=False) -> None:
        """Converts the board grid to a dictionary using grid_to_dict() and uses search.util.print_board() to print
        a visual representation of the board state."""
        print_dict = self.grid_to_dict()
        print_board(print_dict, compact=compact)

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

    def set_winner_terminal(self):
        # analyse remaining tokens
        _WHAT_BEATS = {"r": "p", "p": "s", "s": "r"}
        up_throws = 9 - self.thrown["upper"]
        up_tokens = [s.lower() for (s,v) in self.upper_pieces.items() if len(v) > 0]
        up_symset = set(up_tokens)
        lo_throws = 9 - self.thrown["lower"]
        lo_tokens = [s.lower() for (s,v) in self.lower_pieces.items() if len(v) > 0]
        lo_symset = set(lo_tokens)
        up_invinc = [
            s for s in up_symset
            if (lo_throws == 0) and (_WHAT_BEATS[s] not in lo_symset)
        ]
        lo_invinc = [
            s for s in lo_symset
            if (up_throws == 0) and (_WHAT_BEATS[s] not in up_symset)
        ]
        up_notoks = (up_throws == 0) and (len(up_tokens) == 0)
        lo_notoks = (lo_throws == 0) and (len(lo_tokens) == 0)
        up_onetok = (up_throws == 0) and (len(up_tokens) == 1)
        lo_onetok = (lo_throws == 0) and (len(lo_tokens) == 1)

        # condition 1: one player has no remaining throws or tokens
        if up_notoks and lo_notoks:
            #draw: no remaining tokens or throws
            self.terminal = True
            self.winner = None
            return
        if up_notoks:
            #winner: lower
            self.terminal = True
            self.winner = 'lower'
            return
        if lo_notoks:
            # winner: upper
            self.terminal = True
            self.winner = 'upper'
            return

        # condition 2: both players have an invincible token
        if up_invinc and lo_invinc:
            #draw: both players have an invincible token
            self.terminal = True
            self.winner = None
            return

        # condition 3: one player has an invincible token, the other has
        #              only one token remaining (not invincible by 2)
        if up_invinc and lo_onetok:
            # winner: upper
            self.terminal = True
            self.winner = 'upper'
            return
        if lo_invinc and up_onetok:
            # winner: lower
            self.terminal = True
            self.winner = 'lower'
            return

        # TODO: Impletment this
        # condition 4: the same state has occurred for a 3rd time
        #if self.history[state] >= 3:
        #    self.result = "draw: same game state occurred for 3rd time"
        #    return

        # condition 5: the players have had their 360th turn without end
        if self.n_turns >= 360:
            # draw: maximum number of turns reached
            self.terminal = True
            self.winner = None
            return

        return

    def apply_move(self, move):
        """Applies a move. Moves piece regardless of whether move is valid."""
        type, p1, p2 = move
        if self.turn == 'lower':
            tokens = ('r','p','s')
            pieces = self.lower_pieces
        else:
            tokens = ('R','P','S')
            pieces = self.upper_pieces
        if type == 'SLIDE' or type == 'SWING':
            # We dont get told which token is moving, just where its coming and going from, to work out which one it is, try all 3 and catch errors, t2 is true token
            t2 = None
            #print("trying to remove on of", tokens, "from", self.grid[centered_to_array_coord(p1,self.radius)], "@", p1)
            #print("on board")
            #self.print_grid()
            for t in tokens:
                try:
                    self.grid[centered_to_array_coord(p1,self.radius)].remove(t)
                    self.grid[centered_to_array_coord(p2,self.radius)].append(t)
                    t2 = t
                except ValueError:
                    pass
            try:
                pieces[t2].remove(p1)
                pieces[t2].append(p2)
            except KeyError:
                print(pieces,t2,p1)
                time.sleep(20)
        elif type == 'THROW':
            if self.turn == 'upper':
                self.grid[centered_to_array_coord(p2,self.radius)].append(p1.upper())
                pieces[p1.upper()].append(p2)
            elif self.turn == 'lower':
                self.grid[centered_to_array_coord(p2,self.radius)].append(p1.lower())
                pieces[p1.lower()].append(p2)

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

    def generate_moves(self) -> list:
        """Generates all possible moves for one player."""
        moves = []
        if self.turn == 'upper':
            pieces = self.upper_pieces
        else:
            pieces = self.lower_pieces
        for identifier in pieces:
            for from_tile in pieces[identifier]:
                # Cant use sets since multiple identical pieces can exist on the same coordinate
                slide_moves = [("SLIDE", from_tile, to_tile) for to_tile in
                               get_valid_slides(from_tile, self.radius)]
                swing_moves = [("SWING", from_tile, to_tile) for to_tile in
                               get_valid_swings(from_tile, identifier, self.grid, self.radius)]
                for i in (*slide_moves, *swing_moves): moves.append(i)
        for i in self.get_valid_throws(): moves.append(i)
        return moves

    def get_valid_throws(self):
        """ Returns a list of valid throws for the current turn and player to move"""
        if self.thrown[self.turn] >= 2*self.radius - 1:
            return []
        throws = []
        if self.turn == 'upper':
            flip = 1
            identifiers = ('R', 'P', 'S')
        else:
            flip = -1
            identifiers = ('r', 'p', 's')
        for row in range(1, self.thrown[self.turn] + 2):
            candidate_hexes = [(flip*(self.radius-row), i) for i in range(-(self.radius - 1),self.radius)]
            valid_hexes = [candidate_hexes[i] for i in range(len(candidate_hexes)) if valid_centered_hex(candidate_hexes[i],self.radius)]
            for hex in valid_hexes:
                for i in [("THROW",j.lower(),hex) for j in identifiers]: throws.append(i)
        return throws

    def find_random_child(self):
        if self.terminal:
            return None  # If the game is finished then no moves can be made
        children = self.find_children()
        return random.choice(children)

    def reward(self):
        if not self.terminal:
            raise RuntimeError("reward called on nonterminal board")
        if self.winner is self.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError("reward called on unreachable board ")
        if self.turn is (not self.winner):
            return 0  # Your opponent has just won. Bad.
        if self.winner is None:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def is_terminal(board):
        return board.terminal

    def __eq__(self, other):
        return np.array_equal(self.grid, other.grid)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        string = ''
        for ix, iy in np.ndindex(self.grid.shape):
            string += str(sorted(self.grid[ix, iy]))
        return string