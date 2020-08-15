import unittest

import numpy as np
from researcher import *

class TestRecordManagement(unittest.TestCase):
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

class TestResults(unittest.TestCase):
    def setUp(self):
        self.res = load_experiment("tests/data/", "example_record.json").results
        self.res_multi_epoch = load_experiment("tests/data/", "example_epoch_record.json").results
    
    def test_correctly_gathers_metric(self):
        mses = self.res.get_metric("mse")   

        self.assertEqual(len(mses), 5)
        self.assertEqual(len(mses[0]), 1)
        self.assertEqual(len(mses[1]), 1)
        self.assertEqual(len(mses[2]), 1)
        self.assertEqual(len(mses[3]), 1)
        self.assertEqual(len(mses[4]), 1)

        mses = self.res_multi_epoch.get_metric("mse")   

        self.assertEqual(len(mses), 5)
        self.assertEqual(len(mses[0]), 2)
        self.assertEqual(len(mses[1]), 2)
        self.assertEqual(len(mses[2]), 2)
        self.assertEqual(len(mses[3]), 2)
        self.assertEqual(len(mses[4]), 2)
