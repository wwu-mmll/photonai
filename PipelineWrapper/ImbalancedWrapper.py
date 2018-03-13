from sklearn.base import BaseEstimator, TransformerMixin


class ImbalancedDataTransform(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, imbalance_type: str, method_name: str, **kwargs):
        self.imbalance_type = imbalance_type.lower().strip("_")
        self.method_name = method_name


        ''' Possible values for method_name: 
        imbalance_type = OVERSAMPLING: 
            - RandomOverSampler
            - SMOTE
            - ADASYN
            
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
        '''


        if imbalance_type == "oversampling":
            home = "over_sampling"
        elif imbalance_type == "undersampling":
            home = "under_sampling"
        elif imbalance_type =="combine" or imbalance_type =="combination":
            home = "combine"
        else:
            raise Exception("Imbalance Type not found. Can be oversampling, undersampling or combine")

        # Todo: Try Catch Class Not Found Exception

        desired_class_home = "imblearn." + home
        desired_class_name = method_name

        imported_module = __import__(desired_class_home, globals(), locals(), desired_class_name, 0)
        desired_class = getattr(imported_module, desired_class_name)

        self.method = desired_class(**kwargs)

        self.x_transformed = None
        self.y_transformed = None


    def fit(self, X, y):

        # ATTENTION: Works only if fit is called before transform!!!
        self.x_transformed, self.y_transformed = self.method.fit_sample(X, y)

    def transform(self, X, y):
        return self.x_transformed, self.y_transformed
