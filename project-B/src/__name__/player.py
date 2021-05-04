from __name__.search.miniBoard import Board
from __name__.monte_carlo_tree_search import MCTS, Node
import random


class Player:
    def __init__(self, player):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        # put your code here
        # Initialise the board
        self.tree = MCTS()
        self.game_board = Board(radius=5)
        self.identity = player
        self.game_board.turn = self.identity

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # put your code here
        self.game_board.turn = self.identity
        for i in range(5):
            print(f"{i}/5")
            self.tree.do_rollout(self.game_board)
        self.game_board = self.tree.choose(self.game_board)
        move = self.game_board.last_action

        return move

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        # put your code here
        self.game_board.apply_move(opponent_action)
        self.game_board.n_turns += 1
        if self.game_board.turn == 'upper':
            self.game_board.turn = 'lower'
        else:
            self.game_board.turn = 'upper'



