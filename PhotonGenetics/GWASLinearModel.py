"""
#-----------------------------------------------------------------------#
#                            PHOTON Genetics                            #
#-----------------------------------------------------------------------#
#               GWAS Linear Model for Feature Selection                 #
#-----------------------------------------------------------------------#

__author__ = "Nils Winter"
__copyright__ = "Copyright 2018, Westfälische Wilhelms-Universität Münster "
__license__ = "GPLv3"
__version__ = "2018-01-10"
__email__ = "nils.r.winter@uni-muenster.de"
"""

from photon_core.Logging.Logger import Logger
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

class GWASLinearModel(BaseEstimator):
    """GWAS Linear Model for Feature Selection

    Compute a linear model to test the association between genotype and phenotype. This approach is similar to a standard
    Genome-Wide Association Study.

    Include additional variables as covariates in the linear model to control for i.e. stratification problems. This
    script runs a principal components analysis, first.

    Parameter
    ------------
    some_variable: int

    Examples
    --------
    >>> from photon_core.PhotonGenetics.GWASLinearModel import GWASLinearModel
    See also
    --------
    GWASLogisticModel
        blabla
    TrendTest: Genetic trend test.
    """

    def __init__(self, n_pca_covariates=None):
        self.n_pca_covariates = n_pca_covariates
        self.pca_components = None
        self.my_pca = None

    def fit(self, X, y):
        # do PCA first
        if self.n_pca_covariates:
            self.my_pca = PCA(self.n_pca_covariates)
            self.pca_components = self.my_pca.fit_transform(X)

        # compute linear model for every SNP, use parallelization

        return self

    def transform(self, X, y=None):
        return

    def linear_model(self, x, y, covs):

        pass

    def linear_model(self, snps, phenotypes, cofactors=None):
        lm = LinearModel(phenotypes)
        if cofactors:
            for cofactor in cofactors:
                lm.add_factor(cofactor)
        Logger.verbose("Running a standard linear model")
        res = lm.fast_f_test(snps)
        return res
