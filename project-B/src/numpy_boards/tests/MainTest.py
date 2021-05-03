import unittest
import json
import os
from search.main import main
from search.Board import Board
from ddt import ddt, data


@ddt
class TestMain(unittest.TestCase):

    @data(*[i for i in os.listdir() if i.endswith('.json')])
    def test_all_files(self, value):
        with open(value) as f:
            board_data = json.load(f)
        self.assertTrue(main(data=board_data))

    # @data(*[i for i in os.listdir() if i.endswith('.json')])
    # def test_hash(self, value):
    #     with open(value) as f:
    #         board_data = json.load(f)

    #     board1 = Board(radius=5)
    #     board1.populate_grid(board_data)
    #     print("Board 1")
    #     board1.print_grid()
    #     board2 = Board(radius=5)
    #     board2.populate_grid(board_data)
    #     print("Board 2")
    #     board2.print_grid()

    #     self.assertEqual(hash(board1), hash(board2))


if __name__ == '__main__':
    unittest.main()
