from numpy_boards.search.Board import Board
from numpy_boards.search.monte_carlo_tree_search import MCTS


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
        for i in range(5):
            print(f"{i}/5")
            self.tree.do_rollout(self.game_board)
        next_board = self.tree.choose(self.game_board)

        return next_board.moves[-1]

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        # put your code here
        self.game_board.apply_move(opponent_action, player_action)
        print(self.game_board.remaining_throws)

        self.game_board.move_n += 2  # Applied 2 moves to the game board
