from sklearn.base import BaseEstimator, TransformerMixin
from photonai.photonlogger.logger import logger

try:
    from imblearn import over_sampling, under_sampling, combine
    __found__ = True
except ModuleNotFoundError:
    __found__ = False

class ImbalancedDataTransform(BaseEstimator, TransformerMixin):
    """
    Applies the chosen strategy to the data in order to handle the imbalance in the data.
    Instantiates the strategy filter object according to the name given as string.
    Underlying architecture: Imbalanced-Learning
    More infomration on: https://imbalanced-learn.readthedocs.io/en/stable/api.html
    """
    _estimator_type = "transformer"

    IMBALANCED_DICT = {
        'oversampling': ["ADASYN",
                         "BorderlineSMOTE",
                         "KMeansSMOTE",
                         "RandomOverSampler",
                         "SMOTE",
                         "SMOTENC",
                         "SVMSMOTE"],
        'undersampling': ["AllKNN",
                          "ClusterCentroids",
                          "CondensedNearestNeighbour",
                          "EditedNearestNeighbours",
                          "InstanceHardnessThreshold",
                          "NearMiss",
                          "NeighbourhoodCleaningRule",
                          "OneSidedSelection",
                          "TomekLinks",
                          "RandomUnderSampler",
                          "RepeatedEditedNearestNeighbours"],
        'combine': ["SMOTEENN", "SMOTETomek"],
    }

    def __init__(self, method_name: str='RandomUnderSampler', **kwargs):
        """
        Instantiates an object that transforms the data into balanced groups according to the given method

        Possible values for method_name:
        imbalance_type = OVERSAMPLING:
            - ADASYN
            - BorderlineSMOTE
            - KMeansSMOTE
            - RandomOverSampler
            - SMOTE
            - SMOTENC
            - SVMSMOTE

        imbalance_type = UNDERSAMPLING:
            - ClusterCentroids,
            - RandomUnderSampler,
            - NearMiss,
            - InstanceHardnessThreshold,
            - CondensedNearestNeighbour,
            - EditedNearestNeighbours,
            - RepeatedEditedNearestNeighbours,
            - AllKNN,
            - NeighbourhoodCleaningRule,
            - OneSidedSelection

        imbalance_type = COMBINE:
            - SMOTEENN,
            - SMOTETomek

        :param method_name: which imbalanced strategy to use
        :type method_name: str
        :param kwargs: any parameters to pass to the imbalance strategy object
        :type kwargs:  dict
        """

        if not __found__:
            raise ModuleNotFoundError("Module imblearn not found or not installed as expected. "
                                      "Please install the requirements.txt in PHOTON main folder.")

        self.method_name = method_name
        self.needs_y = True

        imbalance_type = ''
        for group, possible_strategies in ImbalancedDataTransform.IMBALANCED_DICT.items():
            if self.method_name in possible_strategies:
                imbalance_type = group

        if imbalance_type == "oversampling":
            home = over_sampling
        elif imbalance_type == "undersampling":
            home = under_sampling
        elif imbalance_type == "combine" or imbalance_type == "combination":
            home = combine
        else:
            msg = "Imbalance Type not found. Can be oversampling, undersampling or combine."
            msg += "oversampling: method_name one of "+str(self.IMBALANCED_DICT["oversampling"])
            msg += "undersampling: method_name one of "+str(self.IMBALANCED_DICT["undersampling"])
            msg += "combine: method_name one of " + str(self.IMBALANCED_DICT["combine"])
            logger.error(msg)
            raise ValueError(msg)

        desired_class = getattr(home, method_name)
        self.method = desired_class(**kwargs)

        self.x_transformed = None
        self.y_transformed = None

    def fit_transform(self, X, y):
        self.x_transformed, self.y_transformed = self.method.fit_sample(X, y)
        return self.x_transformed, self.y_transformed

    #  define an alias for imblearn consistency
    fit_sample = fit_transform
    fit_resample = fit_transform

    def fit(self, X, y, **kwargs):
        logger.warn("Please use 'fit_transform' in all cases. Store results in x_transformed and y_transformed.")
        self.x_transformed, self.y_transformed = self.method.fit_sample(X, y)
        return

    def transform(self, X, y=None, **kwargs):
        logger.warn("Please use 'fit_transform' in all cases.")
        return

