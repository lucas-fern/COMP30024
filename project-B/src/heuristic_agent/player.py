from heuristic_agent.search.Board import Board
from copy import deepcopy
import math


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
        self.player_num = Board.PLAYER_NUMS[player]
        Board.PLAYER_ID = self.player_num
        # Model the game sequentially with our player always starting
        self.game_board = Board(move_n=0, current_player_n=self.player_num, remaining_throws={0: 9, 1: 9})

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # put your code here

        children = self.game_board.find_top_n_children()
        best_move = None
        # best score is either positive or negative inf depending on whether we maximise or minimise
        best_score = -math.inf
        for child in children:
            # Use negamax to find the score for the child node
            score = child.find_NM_score(depth=2, alpha=-math.inf, beta=math.inf, player_num=child.current_player_n)
            # print(child.moves, score)
            # update best move
            if score > best_score:
                # print(best_score,best_move)
                best_move, best_score = child.moves[-1], score
        return best_move
        # old

        children = self.game_board.find_children()
        best_child = max(children, key=lambda x: x.heuristic)

        return best_child.moves[-1]

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        # put your code here
        # TODO make the game board a new object each time so that the hash value of old boards doesn't change
        assert self.player_num == Board.PLAYER_ID, "update called on opponents board"
        new_game_board = deepcopy(self.game_board)
        self.game_board = new_game_board
        self.game_board.apply_move(opponent_action, player_action)
        print(f'Heuristic agent heuristic: {self.game_board.heuristic_ZS}')
        self.game_board.move_n += 2  # Applied 2 moves to the game board
