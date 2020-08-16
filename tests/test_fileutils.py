import unittest

import numpy as np
from researcher.fileutils import *

class TestFileUtils(unittest.TestCase):
    def setUp(self):
        self.data_path = "tests/data/"

    def test_does_not_mutate_params(self):
        params = {"a": 4, "b": 8, "c": [5, 6, 7, ]}
        expected = {"a": 4, "b": 8, "c": [5, 6, 7, ]}

        save_experiment(self.data_path, "somename", params, {"loss": [0.1, 0.4, 0.231]})

        self.assertDictEqual(params, expected)

    def test_handles_floats(self):
        params = {"a": 4, "b": 8, "c": [5, 6, 7, ]}
        expected = {"a": 4, "b": 8, "c": [5, 6, 7, ]}

        save_experiment(self.data_path, "somename", params, {"loss": [np.float32(0.1), 0.4, 0.231]})

    def test_saves_correctly(self):
        params = {"a": 4, "b": 8, "c": [5, 6, 7, ]}
        expected = {"a": 4, "b": 8, "c": [5, 6, 7, ], "results": {"loss": [0.1, 0.4, 0.231]}}

        save_experiment(self.data_path, "somename", params, {"loss": [0.1, 0.4, 0.231]})

        with open(self.data_path + "somename.json") as f:
            saved = json.load(f)

        self.assertDictEqual(saved, expected)