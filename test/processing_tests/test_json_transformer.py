import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold

from photonai.base.json_transformer import JsonTransformer
from photonai.base import Hyperpipe, Preprocessing
from photonai.optimization import Categorical
from photonai.optimization.hyperparameters import IntegerRange, FloatRange
from photonai.base import Stack, Switch, Branch, PipelineElement
from test.base_tests.test_photon_elements import elements_to_dict


class JsonTransformerTest(unittest.TestCase):

    def setUp(self):
        """
        Set up for Scorer Tests.
        """
        self.maxDiff = None

    def test_branch_in_branch(self):
        """
        Test for deep Pipeline.
        """

        my_pipe = Hyperpipe('basic_stacking',
                            optimizer='grid_search',
                            metrics=['accuracy', 'precision', 'recall'],
                            best_config_metric='f1_score',
                            outer_cv=KFold(n_splits=2),
                            inner_cv=KFold(n_splits=3),
                            verbosity=1,
                            cache_folder="./cache/",
                            project_folder='./tmp/')

        # BRANCH WITH QUANTILTRANSFORMER AND DECISIONTREECLASSIFIER
        tree_qua_branch = Branch('tree_branch')
        tree_qua_branch += PipelineElement('QuantileTransformer')
        tree_qua_branch += PipelineElement('DecisionTreeClassifier',
                                           {'min_samples_split': IntegerRange(2, 4)},
                                           criterion='gini')

        # BRANCH WITH MinMaxScaler AND DecisionTreeClassifier
        svm_mima_branch = Branch('svm_branch')
        svm_mima_branch += PipelineElement('MinMaxScaler')
        svm_mima_branch += PipelineElement('SVC',
                                           {'kernel': ['rbf', 'linear'],  # Categorical(['rbf', 'linear']),
                                            'C': IntegerRange(0.01, 2.0)},
                                           gamma='auto')

        # BRANCH WITH StandardScaler AND KNeighborsClassifier
        knn_sta_branch = Branch('neighbour_branch')
        knn_sta_branch += PipelineElement('StandardScaler')
        knn_sta_branch += PipelineElement('KNeighborsClassifier')

        # voting = True to mean the result of every branch
        my_pipe += Stack('final_stack', [tree_qua_branch, svm_mima_branch, knn_sta_branch])
        my_pipe += PipelineElement('LogisticRegression', solver='lbfgs')

        json_transformer = JsonTransformer()
        pipe_json = json_transformer.create_json(my_pipe)
        my_pipe_reload = json_transformer.from_json(pipe_json)
        pipe_json_reload = pipe_json = json_transformer.create_json(my_pipe_reload)
        self.assertEqual(pipe_json, pipe_json_reload)

    def test_class_with_data_preproc(self):
        """
        Test for simple pipeline with data.
        """
        X, y = load_breast_cancer(return_X_y=True)

        # DESIGN YOUR PIPELINE
        my_pipe = Hyperpipe('basic_svm_pipe',
                            optimizer='grid_search',
                            metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                            best_config_metric='accuracy',
                            use_test_set=False,
                            outer_cv=KFold(n_splits=2),
                            inner_cv=KFold(n_splits=3),
                            verbosity=0,
                            random_seed=42)

        preprocessing = Preprocessing()
        preprocessing += PipelineElement("LabelEncoder")
        my_pipe += preprocessing

        # ADD ELEMENTS TO YOUR PIPELINE
        # first normalize all features
        my_pipe.add(PipelineElement('StandardScaler'))

        # then do feature selection using a PCA,
        my_pipe += PipelineElement('PCA',
                                   hyperparameters={'n_components': IntegerRange(10, 12)},
                                   test_disabled=True)

        # engage and optimize the good old SVM for Classification
        my_pipe += PipelineElement('SVC',
                                   hyperparameters={'kernel': Categorical(['rbf', 'linear'])
                                                    },
                                   C=2,
                                   gamma='scale')

        # NOW TRAIN YOUR PIPELINE
        my_pipe.fit(X, y)

        json_transformer = JsonTransformer()

        pipe_json = json_transformer.create_json(my_pipe)
        _ = elements_to_dict(my_pipe.copy_me())
        my_pipe_reload = json_transformer.from_json(pipe_json)
        pipe_json_reload = pipe_json = json_transformer.create_json(my_pipe_reload)

        self.assertEqual(pipe_json, pipe_json_reload)
        my_pipe_reload.fit(X, y)

        self.assertDictEqual(my_pipe.best_config, my_pipe_reload.best_config)

        self.assertDictEqual(elements_to_dict(my_pipe.copy_me()), elements_to_dict(my_pipe_reload.copy_me()))

    def test_class_with_data_02(self):
        """
        Test for Pipeline with data.
        """
        # DESIGN YOUR PIPELINE
        my_pipe = Hyperpipe(name='Estimator_pipe',
                            optimizer='grid_search',
                            metrics=['balanced_accuracy'],
                            best_config_metric='balanced_accuracy',
                            outer_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                            inner_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                            project_folder='./tmp/',
                            random_seed=42)

        # ADD ELEMENTS TO YOUR PIPELINE
        # first normalize all features
        my_pipe += PipelineElement('StandardScaler')

        # some feature selection
        my_pipe += PipelineElement('LassoFeatureSelection',
                                   hyperparameters={'percentile': FloatRange(start=0.1, step=0.1, stop=0.7,
                                                                             range_type='range'),
                                                    'alpha': FloatRange(0.5, 1)},
                                   test_disabled=True)

        # add imbalanced group handling
        my_pipe += PipelineElement('ImbalancedDataTransformer', method_name='SMOTE', test_disabled=False)

        # setup estimator stack
        est_stack = Stack(name='classifier_stack')
        clf_list = ['RandomForestClassifier', 'LinearSVC', 'NuSVC', "SVC", "MLPClassifier",
                    "KNeighborsClassifier", "Lasso", "PassiveAggressiveClassifier", "LogisticRegression",
                    "Perceptron", "RidgeClassifier", "SGDClassifier", "GaussianProcessClassifier",
                    "AdaBoostClassifier", "BaggingClassifier", "GradientBoostingClassifier"]

        for clf in clf_list:
            est_stack += PipelineElement(clf)
        my_pipe += est_stack

        my_pipe += PipelineElement('PhotonVotingClassifier')

        json_transformer = JsonTransformer()
        pipe_json = json_transformer.create_json(my_pipe)
        my_pipe_reload = json_transformer.from_json(pipe_json)

        self.assertDictEqual(elements_to_dict(my_pipe.copy_me()), elements_to_dict(my_pipe_reload.copy_me()))

    def test_class_switch(self):
        """
        Test for Pipeline with data.
        """
        my_pipe = Hyperpipe('basic_switch_pipe',
                            optimizer='random_grid_search',
                            optimizer_params={'n_configurations': 15},
                            metrics=['accuracy', 'precision', 'recall'],
                            best_config_metric='accuracy',
                            outer_cv=KFold(n_splits=3),
                            inner_cv=KFold(n_splits=5),
                            verbosity=0,
                            project_folder='./tmp/')

        # Transformer Switch
        my_pipe += Switch('TransformerSwitch',
                          [PipelineElement('StandardScaler'),
                           PipelineElement('PCA', test_disabled=True)])

        # Estimator Switch
        svm = PipelineElement('SVC',
                              hyperparameters={'kernel': ['rbf', 'linear']})

        tree = PipelineElement('DecisionTreeClassifier',
                               hyperparameters={'min_samples_split': IntegerRange(2, 5),
                                                'min_samples_leaf': IntegerRange(1, 5),
                                                'criterion': ['gini', 'entropy']})

        my_pipe += Switch('EstimatorSwitch', [svm, tree])

        json_transformer = JsonTransformer()

        pipe_json = json_transformer.create_json(my_pipe)
        my_pipe_reload = json_transformer.from_json(pipe_json)

        self.assertDictEqual(elements_to_dict(my_pipe.copy_me()), elements_to_dict(my_pipe_reload.copy_me()))
