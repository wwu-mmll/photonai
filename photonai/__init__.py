"""
PHOTONAI
is a rapid prototyping framework enabling (not so experienced) users to build, train, optimize, evaluate,
and share even complex machine learning pipelines with very high efficiency.

PHOTONAI is an object-oriented python framework designed to leave you deciding the important things and automatizing the rest:
 By treating each pipeline element as a building block, we create a system in which the user can select and combine
 processing steps, adapt their arrangement or stack them in more advanced pipeline layouts.

PHOTONAI gives you an easy way of setting up a full stack machine learning pipeline including nested cross-validation and hyperparameter search.
After PHOTON has found the best configuration for your model, it offers a convenient possibility to explore the analyzed hyperparameter space.
It also enables you to persist and load your optimal model, including all preprocessing elements, with only one line of code.

"""
from .version import __version__

from .base import Hyperpipe, OutputSettings, Stack, Switch, Branch, PipelineElement, ParallelBranch, \
    PhotonRegistry, DataFilter, CallbackElement, Preprocessing
from .optimization import FloatRange, IntegerRange, Categorical, MinimumPerformanceConstraint, \
    BestPerformanceConstraint, DummyPerformanceConstraint, BooleanSwitch
from .base.json_transformer import JsonTransformer
from .processing.permutation_test import PermutationTest
from .processing import ResultsHandler
