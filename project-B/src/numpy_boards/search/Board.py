import numpy as np
import random
from copy import deepcopy
from itertools import product
from numpy_boards.search.board_util import *
from numpy_boards.search.monte_carlo_tree_search import Node


class Board(Node):  # Putting Node in the brackets because this Inherits from Node class. Forces method implementation.
    N_HEXES = 61
    HEXES_PER_ROW = (5, 6, 7, 8, 9, 8, 7, 6, 5)
    HEX_OPTIONS = 6  # 3 options for the lower tokens, 3 for the upper tokens
    TOKEN_COLS = {
        'r': 0,
        'p': 1,
        's': 2
    }
    IDENTIFIERS = ('r', 'p', 's')
    PLAYER_NUMS = {
        'upper': 0,
        'lower': 1
    }
    PLAYER_ID = None

    def __init__(self, move_n, current_player_n, remaining_throws, board=None, moves=None):
        # Board is represented as a 61 x 6 matrix with counts of tokens on each of the hexes. 0-60 hexes are counted top
        # to bottom, left to right:
        #
        #   R  P  S  r  p  s
        # [[0, 0, 1, 0, 0, 0],
        #  [0, 0, 0, 0, 0, 0],
        #         ...
        #  [0, 0, 0, 0, 2, 0],
        #  [0, 0, 0, 0, 0, 0]]
        # ^ represents one S on hex 0. 2 p's on hex 59
        self.board = board
        if board is None:
            # Use a tiny 8 bit unsigned int (0-64)
            self.board = np.zeros(shape=(Board.N_HEXES, Board.HEX_OPTIONS), dtype=np.uint8)

        self.move_n = move_n
        self.current_player_n = current_player_n  # TODO: make both players operate on the same board

        self.remaining_throws = remaining_throws

        self.moves = moves
        if (moves is not None) and (current_player_n == Board.PLAYER_ID):  # If there are moves, and its our turn
            assert len(moves) == 2, f"Invalid amount of moves for application {moves}."
            player_action, opponent_action = moves

            self.apply_move(opponent_action, player_action)

        self.game_is_over, self.winner = self.get_winner()

    @property
    def turn_n(self):
        return (self.move_n // 2) + 1

    @property
    def next_player_n(self):
        return (self.current_player_n + 1) % 2

    def get_winner(self):
        # analyse remaining tokens
        _WHAT_BEATS = {"r": "p", "p": "s", "s": "r"}
        _TOKENS = np.array(['r', 'p', 's'])
        up_throws = self.remaining_throws[Board.PLAYER_NUMS['upper']]
        # The list of up tokens is any token where the board has any nonzero elements in the column (0-2)
        up_tokens = _TOKENS[np.nonzero(np.any(self.board[:, :3] != 0, axis=0))[0].tolist()]
        up_symset = set(up_tokens)
        lo_throws = self.remaining_throws[Board.PLAYER_NUMS['lower']]
        # Similarly for lower tokens but with the right half of the board array
        lo_tokens = _TOKENS[np.nonzero(np.any(self.board[:, 3:] != 0, axis=0))[0].tolist()]
        lo_symset = set(lo_tokens)
        up_invinc = [
            s for s in up_symset if (lo_throws == 0) and (_WHAT_BEATS[s] not in lo_symset)
        ]
        lo_invinc = [
            s for s in lo_symset if (up_throws == 0) and (_WHAT_BEATS[s] not in up_symset)
        ]
        up_notoks = (up_throws == 0) and (len(up_tokens) == 0)
        lo_notoks = (lo_throws == 0) and (len(lo_tokens) == 0)
        up_onetok = (up_throws == 0) and (len(up_tokens) == 1)
        lo_onetok = (lo_throws == 0) and (len(lo_tokens) == 1)

        # condition 1: one player has no remaining throws or tokens
        if up_notoks and lo_notoks:
            # draw: no remaining tokens or throws
            return True, None
        if up_notoks:
            # winner: lower
            return True, 'lower'
        if lo_notoks:
            # winner: upper
            return True, 'upper'

        # condition 2: both players have an invincible token
        if up_invinc and lo_invinc:
            # draw: both players have an invincible token
            return True, None

        # condition 3: one player has an invincible token, the other has
        #              only one token remaining (not invincible by 2)
        if up_invinc and lo_onetok:
            # winner: upper
            return True, 'upper'
        if lo_invinc and up_onetok:
            # winner: lower
            return True, 'lower'

        # TODO: Implement this
        # condition 4: the same state has occurred for a 3rd time
        # if self.history[state] >= 3:
        #     self.result = "draw: same game state occurred for 3rd time"
        #     return

        # condition 5: the players have had their 360th turn without end
        if self.turn_n >= 360:
            # draw: maximum number of turns reached
            return True, None

        return False, None

    def player_n_pieces(self, n):
        """Returns the columns of the game board corresponding to the player number provided. 0: Upper, 1: Lower"""
        assert (n == 0) or (n == 1), "Invalid player number in column retrieval"

        # Return the first 3 cols if n is 0, otherwise last 3
        return self.board[:, n*3: n*3+3]

    def find_children(self):
        children = set()
        if self.game_is_over:  # If the game is finished then no moves can be made
            return children

        for move in self.generate_moves():
            new_child = self.create_child(move)
            children.add(new_child)

        return children

    def apply_move(self, opponent_action, player_action):
        if Board.PLAYER_ID == 0:
            upper_move, lower_move = player_action, opponent_action
        else:
            upper_move, lower_move = opponent_action, player_action

        for player_n, move in enumerate((upper_move, lower_move)):
            players_pieces = self.player_n_pieces(player_n)
            if move[0] == "THROW":
                self.remaining_throws[player_n] -= 1
                identifier = move[1]
                linear_coord = AXIAL_TO_LINEAR[move[2]]
                # Add a piece to the linear coordinate board, in the column of the relevant piece
                players_pieces[linear_coord, Board.IDENTIFIERS.index(identifier)] += 1

            else:
                linear_from = AXIAL_TO_LINEAR[move[1]]
                linear_to = AXIAL_TO_LINEAR[move[2]]

                # Only one type of token will have an entry in the row.
                # Decrease the number at that index and add it to new
                token_type = players_pieces[linear_from].nonzero()[0]
                players_pieces[linear_from, token_type] -= 1
                players_pieces[linear_to, token_type] += 1

        # Always battle since we are always applying both players moves at once
        self.battle()  # TODO: figure out when to battle and do it with the moves

    def battle(self):
        """Lucas's magnum opus. Battles the tokens on each hex."""
        for hex in self.board:
            kill_vector = np.asarray(([int(not bool(hex[i+1] or hex[(i+4) % 6])) for i in range(3)] * 2),
                                     dtype=np.uint8)
            hex *= kill_vector

    def generate_moves(self) -> list:
        """Generates all possible moves for one player."""
        slide_moves = swing_moves = ()
        player_pieces = self.player_n_pieces(self.current_player_n)
        for i, row in enumerate(player_pieces):
            axial_coord = LINEAR_TO_AXIAL[i]
            if np.any(row):
                slide_moves = [("SLIDE", axial_coord, to_tile) for to_tile in
                               get_valid_slides(axial_coord)]
                swing_moves = [("SWING", axial_coord, to_tile) for to_tile in
                               get_valid_swings(axial_coord, player_pieces)]

        throw_moves = self.get_valid_throws()
        all_moves = [*slide_moves, *swing_moves, *throw_moves]

        return all_moves

    def get_valid_throws(self):
        if self.remaining_throws[self.current_player_n] < 1:
            return []

        # Calculate the number of hexes the player can throw on by summing the number of hexes in each available row
        n_throwable = sum(Board.HEXES_PER_ROW[:10-self.remaining_throws[self.current_player_n]])
        # If the team is Upper, all the throwable tokens are starting from linear coord 0 and counting up. Lower counts
        # back from the end. Tile 3 times for each of the token types to have separate throw actions.
        if self.current_player_n == 0:  # Upper
            throwable_axial = [LINEAR_TO_AXIAL[i] for i in range(n_throwable)]
        else:  # Lower
            throwable_axial = [LINEAR_TO_AXIAL[i] for i in range(Board.N_HEXES-n_throwable, Board.N_HEXES)]

        return [('THROW', identifier, coord) for identifier, coord in product(('r', 'p', 's'), throwable_axial)]

    def find_random_child(self):
        random_move = random.choice(self.generate_moves())

        return self.create_child(random_move)

    def create_child(self, move):
        new_board = np.copy(self.board)
        remaining_throws = deepcopy(self.remaining_throws)
        next_move_n = self.move_n + 1  # TODO: make sure to consider this increments on each players turn

        moves = self.moves + (move,) if (self.moves and len(self.moves) == 1) else (move,)

        return Board(next_move_n, self.next_player_n, remaining_throws, new_board, moves)

    def __deepcopy__(self, memo=None):
        new_board = np.copy(self.board)
        remaining_throws = deepcopy(self.remaining_throws)
        moves = deepcopy(self.moves)

        return Board(self.move_n, self.current_player_n, remaining_throws, new_board, moves)

    def is_terminal(self):
        return self.game_is_over

    def reward(self):
        if self.winner is None:
            return 0.5
        if Board.PLAYER_ID == Board.PLAYER_NUMS['upper']:  # We are playing as upper
            return 1 * int(self.winner == 'upper')
        else:  # We are playing as lower
            return 1 * int(self.winner == 'lower')

    def __hash__(self):
        moves_bytes = bytes(str(self.moves), encoding='utf-8')
        return (self.board.tobytes() + bytes(self.current_player_n) + moves_bytes).__hash__()
        # return hash(str(self))
        # TODO: Figure out what needs to be hashed for MCTS

    def __eq__(node1, node2):
        return np.all(node1.board == node2.board) and \
               (node1.current_player_n == node2.current_player_n) and \
               (node1.moves == node2.moves)
        # return str(node1) == str(node2)
        # TODO: Figure out what needs to be equality checked for MCTS

    def __str__(self):
        string = ''
        for row in self.board:
            string += str(row)
        string += str(self.current_player_n) + str(self.moves)

        return string
