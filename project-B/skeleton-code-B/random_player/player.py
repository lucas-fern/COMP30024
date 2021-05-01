from random_player.search.Board import Board
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
        self.game_board = Board(radius=5)
        self.game_board.identity = player

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # put your code here
        self.game_board.to_move = self.game_board.identity
        moves = self.game_board.generate_moves()
        return random.choice(moves)
    
    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        # put your code here
        self.game_board.apply_move(opponent_action,True)
        self.game_board.apply_move(player_action, False)
        self.game_board.turn += 1
        if player_action[0] == "THROW":
            self.game_board.thrown += 1


