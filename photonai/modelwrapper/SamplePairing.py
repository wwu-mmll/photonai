"""
===========================================================
Project: PHOTON Model Wrapper
===========================================================
Description
-----------
Wrapper for sample pairing

Version
-------
Created:        25-03-2019
Last updated:   25-03-2019


Author
------
Tim Hahn & Nils Winter
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import random
from scipy.spatial.distance import pdist
import math
import itertools


class SamplePairingRegression(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, generator='random_pair', distance_metric=None, draw_limit=10000, rand_seed=True):

        self.needs_covariates = True
        self.needs_y = True
        self.generator = generator
        self.distance_metric = distance_metric
        self.draw_limit = draw_limit
        self.rand_seed = rand_seed

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Generates "new samples" by computing the mean between all or n_draws pairs of existing samples and appends them to X
        The target for each new sample is computed as the mean between the constituent targets
        :param X: data
        :param y: targets (optional)
        :param draw_limit: in case the full number of combinations is > 10k, how many to draw?
        :param rand_seed: sets seed for random sampling of combinations (for reproducibility only)
        :return: X_new: X and X_augmented; (y_new: the correspoding targets)
        """
        # generate combinations
        combs = self._get_combinations(X=X,
                                  generator=self.generator,
                                  distance_metric=self.distance_metric,
                                  draw_limit=self.draw_limit,
                                  rand_seed=self.rand_seed)

        # compute mean over sample pairs
        X_aug = np.empty([len(combs), X.shape[1]])
        i = 0
        for c in combs:
            X_aug[i] = np.mean([X[c[0]], X[c[1]]], axis=0)
            i += 1
        # add augmented samples to existing data
        X_new = np.concatenate((X, X_aug), axis=0)

        # get the corrsponding targets
        if kwargs:
            for name, kwarg in kwargs.items():
                kwarg_new = np.empty(len(combs))
                for i, c in enumerate(combs):
                    kwarg_new[i] = np.mean([kwargs[name][c[0]], kwargs[name][c[1]]])
                kwargs[name] = np.concatenate((np.asarray(kwargs[name]), kwarg_new))

        if y is not None:
            y_aug = np.empty(len(combs))
            i = 0
            for c in combs:
                y_aug[i] = np.mean([y[c[0]], y[c[1]]])
                i += 1
            # add augmented samples to existing data
            y_new = np.concatenate((y, y_aug))
            return X_new, y_new, kwargs
        else:
            return X_new, None, kwargs

    def _stirling(self, n):
        # http://en.wikipedia.org/wiki/Stirling%27s_approximation
        return math.sqrt(2 * math.pi * n) * (n / math.e) ** n

    def _nCr(self, n, r):
        """
        Get number of combinations of r from n items (e.g. n=10, r=2--> pairs from 0 to 10
        :param n: n items to draw from
        :param r: number of items to draw each time
        :return: approximate number of combinations
        """
        try:
            return (self._stirling(n) / self._stirling(r) / self._stirling(n - r) if n > 20 else math.factorial(n) / math.factorial(
                r) / math.factorial(n - r))
        except:
            return -1

    def random_pair_generator(self, items, rand_seed=True):
        """Return an iterator of random pairs from a list of items."""
        # Keep track of already generated pairs
        used_pairs = set()
        random.seed(rand_seed)
        i = 0
        while True:
            if rand_seed:
                random.seed(i)
                i += 1
            pair = random.sample(items, 2)
            # Avoid generating both (1, 2) and (2, 1)
            pair = tuple(sorted(pair))
            if pair not in used_pairs:
                used_pairs.add(pair)
                yield pair

    def nearest_pair_generator(self, X, distance_metric):
        # return most similar pair (from similar to dissimilar)
        i = 0
        # compute distance matrix and get indices
        # res_order = np.argsort(pdist(X, distance_metric))
        from sklearn.preprocessing import StandardScaler
        s = StandardScaler()
        X = s.fit_transform(X)
        res_order = np.argsort(pdist(X, distance_metric))

        inds = np.triu_indices(X.shape[0], k=1)
        while True:
            # get index tuple for the closest draw_limit samples
            pair = (inds[0][res_order[i]], (inds[1][res_order[i]]))
            i += 1
            yield pair

    def _get_combinations(self, X, draw_limit, rand_seed, distance_metric, generator='random_pair'):
        """
        :param X: data array
        :param draw_limit: in case the full number of combinations is > 10k, how many to draw?
        :param rand_seed: sets seed for random sampling of combinations (for reproducibility)
        :param generator: method with which to obtain sample pairs (samples until draw_limit is reached or generator stops)
                            'random_pair': sample randomly from all pairs
                            'nearest_pair': get most similar pairs
        :param distance metric: if generator is 'nearest_pair', this will set the distance metric to obtained similarity
        :return: list of tuples indicating which samples to merge
        """

        items = range(X.shape[0])
        # limit the number of new samples generated if all combinations > draw_limit
        n_combs = self._nCr(n=len(items), r=2)  # get number of possible pairs (combinations of 2) from this data
        if n_combs > draw_limit or n_combs == -1:

            if generator == 'random_pair':
                combs_generator = self.random_pair_generator(items=items, rand_seed=rand_seed)
            elif generator == 'nearest_pair':
                combs_generator = self.nearest_pair_generator(X=X, distance_metric=distance_metric)

            # Get draw_limit sample pairs
            combs = list()
            for i in range(draw_limit):
                combs.append(list(next(combs_generator)))
        else:
            # get all combinations of samples
            combs = list(itertools.combinations(items, 2))
        return combs
