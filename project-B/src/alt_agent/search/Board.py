import numpy as np
import random, math
from copy import deepcopy
from itertools import product
from MCTS.search.board_util import *
from MCTS.search.monte_carlo_tree_search import Node


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

        # A dict containing the amount of throws each player has left. Keys are player numbers.
        self.remaining_throws = remaining_throws

        self.moves = moves  # A tuple of moves which will eventually be actioned on the board.

        # We apply the moves to the board if it is our players turn. This keeps our representation consistent.
        if (moves is not None) and (current_player_n == Board.PLAYER_ID):
            assert len(moves) == 2, f"Invalid amount of moves for application {moves}."
            player_action, opponent_action = moves

            self.apply_move(opponent_action, player_action)

        self.game_is_over, self.winner = self.get_winner()

        # additional attributes for speed-up
        self.up_throws = None
        self.up_tokens = None
        self.up_symset = None
        self.lo_throws = None
        self.lo_tokens = None
        self.lo_symset = None
        self.up_invinc = None
        self.lo_invinc = None

    @property
    def turn_n(self):
        """Gets the RoPaSci360 turn number (1-indexed) based on the amount of moves that have been made."""
        return (self.move_n // 2) + 1

    @property
    def next_player_n(self):
        """Gets the number of the next player based on the board's current player."""
        return (self.current_player_n + 1) % 2

    def find_NM_score(self, depth=0, alpha=-math.inf, beta=math.inf, player_num=None):
        """Performs Negamax, returns value of a node"""
        if depth == 0 or self.game_is_over:
            return self.eval

        value = -math.inf

        for child in self.find_children():
            #print(child.moves)
            value = max(value, child.find_NM_score(
                depth=depth - 1, alpha=-beta, beta=-alpha, player_num=(player_num + 1) % 2))
            alpha = max(alpha, value)
            if alpha >= beta:
                #print("did a thing")
                break

        return -value

    def get_winner(self):
        """Returns a game over status and the winner of the game (if completed)."""
        # analyse remaining tokens
        _WHAT_BEATS = {"r": "p", "p": "s", "s": "r"}
        _TOKENS = np.array(['r', 'p', 's'])
        # Make these properties of the board so that we dont have to calculate them twice
        self.up_throws = self.remaining_throws[Board.PLAYER_NUMS['upper']]
        self.up_tokens = ['r']*np.sum(self.board[:, 0]) + ['p']*np.sum(self.board[:, 1]) + ['s']*np.sum(self.board[:, 2])
        self.up_symset = set(self.up_tokens)
        self.lo_throws = self.remaining_throws[Board.PLAYER_NUMS['lower']]
        self.lo_tokens = ['r']*np.sum(self.board[:, 3]) + ['p']*np.sum(self.board[:, 4]) + ['s']*np.sum(self.board[:, 5])
        self.lo_symset = set(self.lo_tokens)
        self.up_invinc = [
            s for s in self.up_symset if (self.lo_throws == 0) and (_WHAT_BEATS[s] not in self.lo_symset)
        ]
        self.lo_invinc = [
            s for s in self.lo_symset if (self.up_throws == 0) and (_WHAT_BEATS[s] not in self.up_symset)
        ]
        up_notoks = (self.up_throws == 0) and (len(self.up_tokens) == 0)
        lo_notoks = (self.lo_throws == 0) and (len(self.lo_tokens) == 0)
        up_onetok = (self.up_throws == 0) and (len(self.up_tokens) == 1)
        lo_onetok = (self.lo_throws == 0) and (len(self.lo_tokens) == 1)

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
        if self.up_invinc and self.lo_invinc:
            # draw: both players have an invincible token
            return True, None

        # condition 3: one player has an invincible token, the other has
        #              only one token remaining (not invincible by 2)
        if self.up_invinc and lo_onetok:
            # winner: upper
            return True, 'upper'
        if self.lo_invinc and up_onetok:
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

    @property
    def eval(self):
        '''Returns the value of an evaluation function for the current board state, highly positive is good for upper'''
        # Check if game over
        self.game_is_over, self.winner = self.get_winner()
        # Strong reward if won/lost
        if self.game_is_over and self.winner is not None: return 10000*(1 - 2*abs(self.PLAYER_NUMS[self.winner]))
        elif self.game_is_over: return 0
        # Else, favour player with most tokens and throws
        #print(">>>", self.current_player_n)
        #print(len(self.up_tokens), self.up_throws, len(self.lo_tokens), self.lo_throws)

        return (len(self.up_tokens) + self.up_throws - (len(self.lo_tokens) + self.lo_throws))

    def player_n_pieces(self, n):
        """Returns the columns of the game board corresponding to the player number provided. 0: Upper, 1: Lower"""
        assert (n == 0) or (n == 1), "Invalid player number in column retrieval"

        # Return the first 3 cols if n is 0, otherwise last 3
        return self.board[:, n*3: n*3+3]

    def find_children(self):
        """Finds all the valid children boards of a game board."""
        children = set()
        if self.game_is_over:  # If the game is finished then no moves can be made
            return children

        for move in self.generate_moves():
            new_child = self.create_child(move)
            children.add(new_child)

        return children

    def apply_move(self, opponent_action, player_action):
        """Applies a set of two moves to the game board. Does not increment the move_n."""
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
        """Gets all the valid throw moves for the current player on the board. Formatted according to the rules."""
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
        """Picks a random child of the current board."""
        random_move = random.choice(self.generate_moves())

        return self.create_child(random_move)

    def create_child(self, move):
        """Generates a child of the current board with a specific move applied. Leaves it up to the child to determine
        whether to apply the move (based on whether it is our player's turn)"""
        new_board = np.copy(self.board)
        remaining_throws = deepcopy(self.remaining_throws)
        # TODO reconsider where this is placed. Maybe should increment in the apply_moves() function.
        next_move_n = self.move_n + 1  # TODO: make sure to consider this increments on each players turn

        moves = self.moves + (move,) if (self.moves and len(self.moves) == 1) else (move,)

        return Board(next_move_n, self.next_player_n, remaining_throws, new_board, moves)

    def __deepcopy__(self, memo=None):
        """Creates a deep copy of the board."""
        new_board = np.copy(self.board)
        remaining_throws = deepcopy(self.remaining_throws)
        moves = deepcopy(self.moves)

        return Board(self.move_n, self.current_player_n, remaining_throws, new_board, moves)

    def is_terminal(self):
        """Returns true if the board is in a terminal state. Else false."""
        return self.game_is_over

    def reward(self):
        """Gives the reward to our player for a current terminal state."""
        assert self.game_is_over, 'Asked for reward of non-terminal state'

        if self.winner is None:
            return 0.5
        if Board.PLAYER_ID == Board.PLAYER_NUMS['upper']:  # We are playing as upper
            return 1 * int(self.winner == 'upper')
        else:  # We are playing as lower
            return 1 * int(self.winner == 'lower')

    def __hash__(self):
        """Hashes the board based on the contents of the board array, current player, and stored moves."""
        moves_bytes = bytes(str(self.moves), encoding='utf-8')
        return (self.board.tobytes() + bytes(self.current_player_n) + moves_bytes).__hash__()
        # return hash(str(self))
        # TODO: Figure out what needs to be hashed for MCTS

    def __eq__(node1, node2):
        """Checks for equality of boards based on the contents of the board array, current player, and stored moves."""
        return np.all(node1.board == node2.board) and \
               (node1.current_player_n == node2.current_player_n) and \
               (node1.moves == node2.moves)
        # return str(node1) == str(node2)
        # TODO: Figure out what needs to be equality checked for MCTS
