from photonai import RandomSearchOptimizer, IntegerRange, FloatRange, Categorical, PipelineElement, Hyperpipe
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import pearsonr
import numpy as np

# ------------------------------------------------------------------------------------------------------
# Case 1: Using only random search optimizer to generate the config
# ------------------------------------------------------------------------------------------------------


class TVBDummyElement(BaseEstimator, ClassifierMixin):
    # todo: list tvb params with name here!
    def __init__(self, param1=None, param2=None, param3=None):
        # important: every hyperparameter needs to be stored in an instance variable
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    def fit(self, X, y):
        return self

    def predict(self, X):
        return None


tvb_hyperparameters = {'param1': IntegerRange(0, 10),
                       'param2': FloatRange(0, 10, range_type="geomspace"),
                       'param3': Categorical(['uno', 'dos', 'tres'])
                       }

tvb_dummy = PipelineElement.create('tvb_dummy', TVBDummyElement(), hyperparameters=tvb_hyperparameters)

random_search_optimizer = RandomSearchOptimizer(n_configurations=25)
random_search_optimizer.prepare([tvb_dummy], maximize_metric=True)
for next_config in random_search_optimizer.ask:
    print(next_config)


# ------------------------------------------------------------------------------------------------------
# Case 2: Using photonai as Infrastructure
# will iterate each config the random search generator yields and:
# -> start the docker
# -> load the result
# -> compare the result to the ground truth for the specified metrics
# -> (log each config and all results)
# ------------------------------------------------------------------------------------------------------

class OnePersonNoCV:

    def init_(self):
        pass

    def split(self, X, y=None, groups=None):
        yield np.array([0]), np.array([0])


class TVBStarterElement(TVBDummyElement):

    def __init__(self, docker_container, param1=None, param2=None, param3=None):
        super(TVBStarterElement, self).__init__(param1, param2, param3)
        self.docker_container = docker_container
        self.params = [param1, param2, param3]
        self.outputfile = None

    def fit(self, X, y):
        self.return_value = y

        # # when the code reaches this place, param1, param2, param3 are already set by hyperparameter config.
        # cmd = 'Some TVB Command using param1 {}, param2 {}, param3 {}'.format(*self.params)
        # self.docker_container.exec_run(cmd)
        # # somehow monitor the process until command has finished....
        # self.outputfile = 'Docker Output File'
        # return self

    def predict(self, X):
        return self.return_value

        # outputdata = pd.read_csv(self.outputfile, delimiter=" ")
        # # convert to matrix
        # return outputdata


def score_tvb_output(X, y):
    # of course this works for one person only
    return pearsonr(np.squeeze(X), np.squeeze(y))[0]

# import docker
# https://docker-py.readthedocs.io/en/stable/index.html
# client = docker.from_env()
# tvb_container_id = 'fast-tbv-1'
# docker_container = client.containers.get(tvb_container_id)
docker_container = None

my_first_person_input_file = '/path/to/first/person'
my_first_person_target_file = np.array([0, 0, 1, 1, 0])

setup = Hyperpipe('tvb_optimizer_for_one_person',
                  outer_cv=None,
                  inner_cv=OnePersonNoCV(),  # when using more than one person use shuffle split
                  optimizer='random_search',
                  optimizer_params={'n_configurations': 25},
                  metrics=[('tvb_metric', score_tvb_output)],
                  best_config_metric='tvb_metric',
                  project_folder='./photonai_output_person_1',
                  allow_multidim_targets=True,
                  use_test_set=False,
                  verbosity=2)

setup += PipelineElement.create('tvb_item', TVBStarterElement(docker_container), hyperparameters=tvb_hyperparameters)
setup.fit([my_first_person_input_file], [my_first_person_target_file])




