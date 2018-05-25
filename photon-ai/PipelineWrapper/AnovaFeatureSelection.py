from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats


class AnovaFeatureSelection(BaseEstimator, ClassifierMixin):

    def __init__(self, p_threshold=0.5):
        self.p_threshold = p_threshold

    def fit(self, data, targets):
        return self

    def transform(self, data, targets):

        snp_in = []
        for snpInd in range(data.shape[1]):
            a = targets[data[:, snpInd] == 1]
            b = targets[data[:, snpInd] == 2]
            c = targets[data[:, snpInd] == 3]

            f, p = stats.f_oneway(a, b, c)

            if p < self.p_threshold:
                # print('One-way ANOVA - snp_name ' + snp_names[snpInd])
                # print('=============')
                # print('F value:', f)
                # print('P value:', p, '\n')
                snp_in.append(snpInd)

        if len(snp_in) <= 1:
            return []
        return data[:, snp_in]
