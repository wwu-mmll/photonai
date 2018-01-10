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

class GWASLinearModel(BaseEstimator):
    """GWAS Linear Model for Feature Selection

    Compute a linear model to test the association between genotype and phenotype. This approach is similar to a standard
    Genome-Wide Association Study.

    Include additional variables as covariates in the linear model to control for i.e. stratification problems.

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

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return