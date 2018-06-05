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

#from photon_core.photonlogger.Logger import Logger
from sklearn.base import BaseEstimator
from sklearn.decomposition import RandomizedPCA
from pygwas.core.linear_models import LinearModel
import scipy as sp
from numpy import linalg

class GWASLinearModel(BaseEstimator):
    """GWAS Linear Model for Feature Selection

    Compute a linear model to test the association between genotype and phenotype. This approach is similar to a standard
    Genome-Wide Association Study.

    Include additional variables as covariates in the linear model to control for i.e. stratification problems. This
    script runs a randomized principal components analysis, first.

    Parameter
    ------------
    some_variable: int

    Examples
    --------
    >>> from photon_core.genetics.GWASLinearModel import GWASLinearModel
    See also
    --------
    GWASLogisticModel
        blabla
    TrendTest: Genetic trend test.
    """

    def __init__(self, n_pca_covariates=None, pca_batch_size=50):
        self.n_pca_covariates = n_pca_covariates
        self.pca_components = None
        self.my_pca = None
        self.pca_batch_size = pca_batch_size

    def fit(self, X, y):
        # do PCA first
        if self.n_pca_covariates:
            self.my_pca = RandomizedPCA(n_components=self.n_pca_covariates)
            self.pca_components = self.my_pca.fit_transform(X)

        # compute linear model for every SNP, use parallelization

        return self

    def transform(self, X, y=None):
        return

    def linear_model(self, X, y, cofactors=None):
        lm = LinearModel(y)
        if cofactors:
            for cofactor in cofactors:
                lm.add_factor(cofactor)
        #Logger.verbose("Running a standard linear model")
        res = lm.fast_f_test(X)
        return res

    def fast_f_test(self, genotype, Z=None, with_betas=False):
        """
        LM implementation
        Single SNPs

        Adapted from https://github.com/timeu/PyGWAS/blob/master/pygwas/core/linear_models.py

        """
        snps = genotype.get_snps_iterator(is_chunked=True)
        dtype = 'single'
        q = 1  # Single SNP is being tested
        p = len(self.X.T) + q
        n = self.n
        n_p = n - p
        num_snps = genotype.num_snps

        h0_X = sp.mat(self.X, dtype=dtype)
        (h0_betas, h0_rss, h0_rank, h0_s) = linalg.lstsq(h0_X, self.Y)
        Y = sp.mat(self.Y - h0_X * h0_betas, dtype=dtype)
        h0_betas = map(float, list(h0_betas))

        if not with_betas:
            (Q, R) = qr_decomp(h0_X)  # Do the QR-decomposition for the Gram-Schmidt process.
            Q = sp.mat(Q, dtype=dtype)
            Q2 = Q * Q.T
            M = sp.mat((sp.eye(n) - Q2), dtype=dtype)
        else:
            betas_list = [h0_betas] * num_snps

        rss_list = sp.repeat(h0_rss, num_snps)
        chunk_size = len(Y)
        i = 0
        for snps_chunk in snps:
            if len(snps_chunk) == 0:
                continue
            snps_chunk = sp.matrix(snps_chunk)
            if with_betas:
                Xs = snps_chunk
            else:
                Xs = sp.mat(snps_chunk, dtype=dtype) * M
            for j in range(len(Xs)):
                if with_betas:
                    (betas, rss, p, sigma) = linalg.lstsq(sp.hstack([h0_X, Xs[j].T]), Y, \
                                                          overwrite_a=True)
                    if not rss:
                        log.debug('No predictability in the marker, moving on...')
                        continue
                    betas_list[i] = map(float, list(betas))
                else:
                    (betas, rss, p, sigma) = linalg.lstsq(Xs[j].T, Y, overwrite_a=True)
                rss_list[i] = rss[0]

                if (i + 1) % (num_snps / 10) == 0:
                    perc = 100.0 * i / num_snps
                    log.info('Performing regression (completed:%d %%)' % perc, extra={'progress': 25 + 80 * perc / 100})
                if not progress_file_writer == None:
                    progress_file_writer.update_progress_bar(
                        task_status='Performing regression (completed: %d %%)' % (100.0 * i / num_snps))
                i += 1

        rss_ratio = h0_rss / rss_list
        var_perc = 1 - 1 / rss_ratio
        f_stats = (rss_ratio - 1) * n_p / float(q)
        p_vals = stats.f.sf(f_stats, q, n_p)

        res_d = {'ps': p_vals, 'f_stats': f_stats, 'rss': rss_list, 'var_perc': var_perc,
                 'h0_rss': h0_rss, 'h0_betas': h0_betas}
        if with_betas:
            res_d['betas'] = betas_list
        return res_d


if __name__ == '__main__':
    print('hey')

