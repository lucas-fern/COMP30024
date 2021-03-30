import unittest
import json
import os
from search.main import main
from ddt import ddt, data


@ddt
class TestMain(unittest.TestCase):

    @data(*[i for i in os.listdir() if i.endswith('.json')])
    def test_all_files(self, value):
        with open(value) as f:
            board_data = json.load(f)

        self.assertTrue(main(data=board_data))


if __name__ == '__main__':
    unittest.main(verbosity=2)
