"""
PHOTON
A Python-based Hyperparameter Optimization Toolbox for Neural Networks designed to accelerate and simplify the construction, training, and evaluation of machine learning models.

PHOTON is an object-oriented python framework for optimizing machine learning pipelines,
 designed to leave you deciding the important things and automatizing the rest.

PHOTON gives you an easy way of setting up a full stack machine learning pipeline including nested cross-validation and hyperparameter search.
After PHOTON has found the best configuration for your model, it offers a convenient possibility to explore the analyzed hyperparameter space.
It also enables you to persist and load your optimal model, including all preprocessing steps, with only one line of code.

"""

from . import base
from . import optimization
from . import validation

from .base import Hyperpipe, PipelineStacking, PipelineBranch, PipelineElement
from .optimization import GridSearchOptimizer, TimeBoxedRandomGridSearchOptimizer, RandomGridSearchOptimizer

from . import investigator
from .investigator import Investigator

from . import configuration
from .configuration.Register import PhotonRegister
__version__ = "0.3.7"


__all__ = ("base",
           "optimization",
           "validation",
           "investigator",
           "Investigator",
           "Hyperpipe",
           "PipelineElement",
           "PipelineBranch",
           "PipelineStacking",
           "GridSearchOptimizer",
           "RandomGridSearchOptimizer",
           "TimeBoxedRandomGridSearchOptimizer",
           "configuration",
           "PhotonRegister"
           )
