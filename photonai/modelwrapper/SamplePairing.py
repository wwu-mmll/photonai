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

import itertools
import math
import random

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from photonai.helper.helper import PhotonDataHelper
from photonai.photonlogger.logger import logger


class SamplePairingBase(BaseEstimator, TransformerMixin):

    @staticmethod
    def _stirling(n):
        # http://en.wikipedia.org/wiki/Stirling%27s_approximation
        return math.sqrt(2 * math.pi * n) * (n / math.e) ** n

    @staticmethod
    def _calculate_number_of_possible_combinations(n, r):
        """
        Get number of combinations of r from n items (e.g. n=10, r=2--> pairs from 0 to 10
        :param n: n items to draw from
        :param r: number of items to draw each time
        :return: approximate number of combinations
        """
        try:
            stirling_n = SamplePairingBase._stirling(n)
            stirling_r = SamplePairingBase._stirling(r)
            stirling_n_r = SamplePairingBase._stirling(n - r)

            if n > 20:
                return stirling_n / stirling_r / stirling_n_r
            else:
                return math.factorial(n) / math.factorial(r) / math.factorial(n - r)
        except:
            return -1

    @staticmethod
    def random_pair_generator(sample_indices, rand_seed, draw_limit):
        """Return a list of random pairs from a list of items."""
        random.seed(rand_seed)

        pairs = list()

        for i in range(draw_limit):
            n_attempts = 0
            while True:
                pair = tuple(sorted(random.sample(sample_indices, 2)))
                if pair not in pairs:
                    pairs.append(pair)
                    break
                elif n_attempts > draw_limit:
                    # prevent while loop from continuing infinitely
                    # stop in case the number of attempts exceeds the draw limit
                    # this is just an arbitrary threshold
                    break
                n_attempts += 1
        return pairs

    @staticmethod
    def nearest_pair_generator(X, distance_metric, draw_limit):
        n_samples = X.shape[0]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        distance_indices = np.argsort(pdist(X, distance_metric))
        triu_indices = np.triu_indices(n_samples, k=1)

        pairs = list()
        for i in range(draw_limit):
            if i > len(distance_indices) - 1:
                break
            else:
                pairs.append((triu_indices[0][distance_indices[i]], (triu_indices[1][distance_indices[i]])))
        return pairs

    def _get_pairs(self, X, draw_limit, rand_seed, distance_metric, generator='random_pair'):
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

        sample_indices = range(X.shape[0])

        # limit the number of new samples generated if all combinations > draw_limit
        n_combinations = self._calculate_number_of_possible_combinations(n=len(sample_indices), r=2)

        if n_combinations > draw_limit or n_combinations == -1:
            if generator == 'random_pair':
                return self.random_pair_generator(sample_indices=sample_indices,
                                                  rand_seed=rand_seed,
                                                  draw_limit=draw_limit)
            elif generator == 'nearest_pair':
                return self.nearest_pair_generator(X=X,
                                                   distance_metric=distance_metric,
                                                   draw_limit=draw_limit)
            else:
                raise NotImplementedError("{} is not supported. Possible options: 'random_pair', 'nearest_pair")
        else:
            # get all combinations of samples
            return list(itertools.combinations(sample_indices, 2))

    def _return_samples(self, X, y, kwargs, generator, distance_metric, draw_limit, rand_seed):
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
        pairs = self._get_pairs(X=X,
                                generator=generator,
                                distance_metric=distance_metric,
                                draw_limit=draw_limit,
                                rand_seed=rand_seed)

        # compute mean over sample pairs
        X_new = np.empty([len(pairs), X.shape[1]])
        i = 0
        for pair in pairs:
            X_new[i] = np.mean([X[pair[0]], X[pair[1]]], axis=0)
            i += 1

        # add augmented samples to existing data
        X_new = np.concatenate((X, X_new), axis=0)

        # get the corresponding targets and kwargs
        if kwargs:
            for name, kwarg in kwargs.items():
                kwarg_new = np.empty(len(pairs))
                for i, pair in enumerate(pairs):
                    kwarg_new[i] = np.mean([kwargs[name][pair[0]], kwargs[name][pair[1]]])
                kwargs[name] = np.concatenate((np.asarray(kwargs[name]), kwarg_new))

        if y is not None:
            y_aug = np.empty(len(pairs))
            i = 0
            for pair in pairs:
                y_aug[i] = np.mean([y[pair[0]], y[pair[1]]])
                i += 1
            # add augmented samples to existing data
            y_new = np.concatenate((y, y_aug))
            return X_new, y_new, kwargs
        else:
            return X_new, None, kwargs


class SamplePairingRegression(SamplePairingBase):
    _estimator_type = "transformer"

    def __init__(self, generator='random_pair', distance_metric='euclidean', draw_limit=10000, random_state=45):

        self.needs_covariates = True
        self.needs_y = True
        self.generator = generator
        self.distance_metric = distance_metric
        self.draw_limit = draw_limit
        self.random_state = random_state

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
        return self._return_samples(X, y, kwargs, self.generator, self.distance_metric, self.draw_limit, self.random_state)


class SamplePairingClassification(SamplePairingBase):
    _estimator_type = "transformer"

    def __init__(self, generator='random_pair', distance_metric='euclidean', draw_limit=10000, random_state=45,
                 balance_classes=True):

        self.needs_covariates = True
        self.needs_y = True
        self.generator = generator
        self.distance_metric = distance_metric
        self.draw_limit = draw_limit
        self.random_state = random_state
        self.balance_classes = balance_classes

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

        logger.debug("Pairing " + str(self.draw_limit) + " samples...")

        # ensure class balance in the training set if balance_classes is True
        unique_classes = np.unique(y)
        n_pairs = list()
        for label in unique_classes:
            if self.balance_classes:
                n_pairs.append(self.draw_limit - np.sum(y == label))
            else:
                n_pairs.append(self.draw_limit)

        # run get_samples for each class independently
        X_extended = list()
        y_extended = list()
        kwargs_extended = dict()

        for label, limit in zip(unique_classes, n_pairs):
            X_new_class, y_new_class, kwargs_new_class = self._return_samples(X[y == label], y[y == label],
                                                                              PhotonDataHelper.index_dict(kwargs,
                                                                                                          y == label),
                                                                              generator=self.generator,
                                                                              distance_metric=self.distance_metric,
                                                                              draw_limit=limit,
                                                                              rand_seed=self.random_state)

            X_extended.extend(X_new_class)
            y_extended.extend(y_new_class)

            # get the corresponding kwargs
            if kwargs:
                kwargs_extended = PhotonDataHelper.join_dictionaries(kwargs_extended, kwargs_new_class)

        return X_extended, y_extended, kwargs_extended
