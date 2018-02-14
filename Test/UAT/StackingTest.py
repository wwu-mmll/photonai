import unittest, random, sklearn
import pandas as pd
from sklearn.model_selection import ShuffleSplit, KFold
from photon_core.Framework.PhotonBase import Hyperpipe, \
    PipelineElement, PipelineStacking, SourceFilter

class CVTestsLocalSearchTrue(unittest.TestCase):
    __thickness = None
    __surface = None
    __y = None

    def setUp(self):
        random.seed(42)
        self.__thickness = pd.read_csv('../EnigmaTestFiles/CorticalMeasuresENIGMA_ThickAvg.csv').iloc[1:]
        self.__surface = pd.read_csv('../EnigmaTestFiles/CorticalMeasuresENIGMA_SurfAvg.csv').iloc[1:]
        y = pd.read_csv('../EnigmaTestFiles/Covariates.csv')
        self.__y = y['Sex']


    def testStacking(self):
        svc_c = [.1, 1]
        svc_kernel = ['rbf', 'linear']

        cv_outer = ShuffleSplit(n_splits=1, test_size=0.2, random_state=3)
        cv_inner = ShuffleSplit(n_splits=1, test_size=0.2, random_state=3)


        # SET UP HYPERPIPES

        # surface pipe
        surface_pipe = Hyperpipe('surface_pipe', optimizer='grid_search',
                               metrics=['accuracy'],
                               inner_cv=cv_inner)

        surface_pipe += PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel})
        # use source filter to select data for stacked hyperpipes
        surface_pipe.filter_element = SourceFilter(0)

        # thickness pipe
        thickness_pipe = Hyperpipe('thickness_pipe', optimizer='grid_search',
                               metrics=['accuracy'],
                               inner_cv=cv_inner)

        thickness_pipe += PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel})
        # use source filter to select data for stacked hyperpipes
        thickness_pipe.filter_element = SourceFilter(1)

        # Mother Pipe
        mother = Hyperpipe('mother', optimizer='grid_search',
                           metrics='accuracy',
                           inner_cv=cv_inner,
                           outer_cv=cv_outer,
                           eval_final_performance=True)

        mother += PipelineStacking('multiple_sources', [surface_pipe, thickness_pipe],voting=False)
        mother += PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel})

        mother.fit([self.__surface, self.__thickness], self.__y)
        print()
