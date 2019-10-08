import unittest

import numpy as np

from photonai.modelwrapper.Voting import PhotonVotingRegressor


class VotingTests(unittest.TestCase):

    def setUp(self):
        self.voter = PhotonVotingRegressor()

    def test_wrong_strategy(self):
        with self.assertRaises(ValueError):
            self.voter.strategy = 'random'

    def test_set_params(self):
        self.voter.set_params(**{'strategy': 'mean'})
        self.assertTrue(self.voter.strategy == 'mean')

    def test_method_application(self):
        for strategy_name, _ in self.voter.STRATEGY_DICT.items():
            self.voter.strategy = strategy_name
            X = np.random.randint(0, 10, (50, 20))
            predicted_X = self.voter.predict(X)
            self.assertTrue(len(predicted_X) == X.shape[0])
            if hasattr(np, strategy_name):
                numpy_method = getattr(np, strategy_name)
                numpy_X = numpy_method(X, axis=1)
                self.assertTrue(np.array_equal(numpy_X, predicted_X))

