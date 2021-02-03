
from photonai.helper.photon_base_test import PhotonBaseTest
from photonai.base import PipelineElement, Hyperpipe, ParallelBranch
from photonai.base.cache_manager import CacheManager
from photonai.base.photon_pipeline import PhotonPipeline

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
import os
import glob
import numpy as np


class MakeSomeStupidNoiseMatrices:

    def __call__(self, nr_items: int = 10, folder="./tmp"):
        item_list = list()
        for i in range(nr_items):
            filename = os.path.join(folder, 'test_{}.npy'.format(i))
            with open(filename, 'wb') as f:
                np.save(f, np.random.random(size=(10, )))
                item_list.append(filename)
        return item_list


class StupidAdditionTransformer(BaseEstimator):

    def __init__(self, voxel_size: int = 2):
        self.voxel_size = voxel_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        # load X
        if isinstance(X, str):
            data = np.load(open(X, 'rb'))
            return data + self.voxel_size
        else:
            data_items = list()
            for item in X:
                data = np.load(open(item, 'rb'))
                data_items.append(data + self.voxel_size)
            data_items = np.stack([i for i in data_items], axis=0)
            return data_items


class CachedPhotonPipelineTestsNeuro(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(CachedPhotonPipelineTestsNeuro, cls).setUpClass()

    def setUp(self) -> None:
        super(CachedPhotonPipelineTestsNeuro, self).setUp()
        m = MakeSomeStupidNoiseMatrices()
        self.X = m(folder=self.tmp_folder_path)
        self.y = np.random.randn(len(self.X))
        self.nr_of_expected_pickles_per_config = len(self.X)

    def test_single_subject_caching(self):

        nb = ParallelBranch("subject_caching_test")
        # increase complexity by adding batching
        nb += PipelineElement.create("ResampleImages", StupidAdditionTransformer(), {},
                                     batch_size=4)

        cache_folder = self.cache_folder_path
        cache_folder = os.path.join(cache_folder, 'subject_caching_test')
        nb.base_element.cache_folder = cache_folder

        def transform_and_check_folder(config, expected_nr_of_files):
            nb.set_params(**config)
            nb.transform(self.X, self.y)
            nr_of_generated_cache_files = len(glob.glob(os.path.join(cache_folder, "*.p")))
            self.assertTrue(nr_of_generated_cache_files == expected_nr_of_files)

        # fit with first config
        # expect one cache file per input file
        transform_and_check_folder({'ResampleImages__voxel_size': 5}, self.nr_of_expected_pickles_per_config)

        # after fitting with second config, we expect two times the number of input files to be in cache
        transform_and_check_folder({'ResampleImages__voxel_size': 10}, 2 * self.nr_of_expected_pickles_per_config)

        # fit with first config again, we expect to not have generate other cache files, because they exist
        transform_and_check_folder({'ResampleImages__voxel_size': 5}, 2 * self.nr_of_expected_pickles_per_config)

        # clean up afterwards
        CacheManager.clear_cache_files(cache_folder)
        CacheManager.clear_cache_files(self.tmp_folder_path, force_all=True)

    def test_combi_from_single_and_group_caching(self):

        # 2. specify cache directories
        cache_folder_base = self.cache_folder_path
        cache_folder_neuro = os.path.join(cache_folder_base, 'subject_caching_test')

        CacheManager.clear_cache_files(cache_folder_base)
        CacheManager.clear_cache_files(cache_folder_neuro)

        # 3. set up Neuro Branch
        nb = ParallelBranch("SubjectCaching", nr_of_processes=3)
        # increase complexity by adding batching
        nb += PipelineElement.create("ResampleImages", StupidAdditionTransformer(), {},
                                     batch_size=4)
        nb.base_element.cache_folder = cache_folder_neuro

        # 4. setup usual pipeline
        ss = PipelineElement("StandardScaler", {})
        pca = PipelineElement("PCA", {'n_components': [3, 10, 50]})
        svm = PipelineElement("SVR", {'kernel': ['rbf', 'linear']})

        pipe = PhotonPipeline([('NeuroBranch', nb),
                               ('StandardScaler', ss),
                               ('PCA', pca),
                               ('SVR', svm)])

        pipe.caching = True
        pipe.fold_id = "12345643463434"
        pipe.cache_folder = cache_folder_base

        def transform_and_check_folder(config, expected_nr_of_files_group, expected_nr_subject):
            pipe.set_params(**config)
            pipe.fit(self.X, self.y)
            nr_of_generated_cache_files = len(glob.glob(os.path.join(cache_folder_base, "*.p")))
            self.assertTrue(nr_of_generated_cache_files == expected_nr_of_files_group)

            nr_of_generated_cache_files_subject = len(glob.glob(os.path.join(cache_folder_neuro, "*.p")))
            self.assertTrue(nr_of_generated_cache_files_subject == expected_nr_subject)

        config1 = {'NeuroBranch__ResampleImages__voxel_size': 5, 'PCA__n_components': 7, 'SVR__C': 2}
        config2 = {'NeuroBranch__ResampleImages__voxel_size': 3, 'PCA__n_components': 4, 'SVR__C': 5}

        # first config we expect to have a cached_file for the standard scaler and the pca
        # and we expect to have two files (one resampler, one brain mask) for each input data
        transform_and_check_folder(config1, 2,  self.nr_of_expected_pickles_per_config)

        # second config we expect to have two cached_file for the standard scaler (one time for 5 voxel input and one
        # time for 3 voxel input) and two files two for the first and second config pcas,
        # and we expect to have 2 * nr of input data for resampler plus one time masker
        transform_and_check_folder(config2, 4, 2 * self.nr_of_expected_pickles_per_config)

        # when we transform with the first config again, nothing should happen
        transform_and_check_folder(config1, 4, 2 * self.nr_of_expected_pickles_per_config)

        # when we transform with an empty config, a new entry for pca and standard scaler should be generated, as well
        # as a new cache item for each input data from the neuro branch for each itemin the neuro branch
        with self.assertRaises(ValueError):
            transform_and_check_folder({}, 6, 4 * self.nr_of_expected_pickles_per_config)

        CacheManager.clear_cache_files(cache_folder_base)
        CacheManager.clear_cache_files(cache_folder_neuro)


class CachedHyperpipeTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(CachedHyperpipeTests, cls).setUpClass()

    def setUp(self) -> None:
        super(CachedHyperpipeTests, self).setUp()
        m = MakeSomeStupidNoiseMatrices()
        self.X = m(folder=self.tmp_folder_path)
        self.y = np.random.randn(len(self.X))
        self.nr_of_expected_pickles_per_config = len(self.X)

    def test_neuro_hyperpipe_parallelized_batched_caching(self):

        cache_path = self.cache_folder_path

        self.hyperpipe = Hyperpipe('complex_case',
                                   inner_cv=KFold(n_splits=5),
                                   outer_cv=KFold(n_splits=3),
                                   optimizer='grid_search',
                                   cache_folder=cache_path,
                                   metrics=['mean_squared_error'],
                                   best_config_metric='mean_squared_error',
                                   project_folder=self.tmp_folder_path)

        nb = ParallelBranch("SubjectCaching", nr_of_processes=1)
        nb += PipelineElement.create("ResampleImages", StupidAdditionTransformer(), {'voxel_size': [3, 5, 10]},
                                     batch_size=4)
        self.hyperpipe += nb

        self.hyperpipe += PipelineElement("StandardScaler", {})
        self.hyperpipe += PipelineElement("PCA", {'n_components': [3, 4]})
        self.hyperpipe += PipelineElement("SVR", {'kernel': ['rbf', 'linear']})

        self.hyperpipe.fit(self.X, self.y)

        # assert cache is empty again
        nr_of_p_files = len(glob.glob(os.path.join(self.hyperpipe.cache_folder, "*.p")))
        print(nr_of_p_files)
        self.assertTrue(nr_of_p_files == 0)
