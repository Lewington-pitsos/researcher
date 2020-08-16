import unittest

from researcher.results import *
from researcher.fileutils import load_experiment

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
