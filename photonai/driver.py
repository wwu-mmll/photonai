import os

# import pixiedust
from photonai.base import PhotonRegistry


__file__ = "kmeans.log"  # theNotebook + ".ipynb"
base_folder = os.path.dirname(os.path.abspath(__file__))
custom_elements_folder = os.path.join(base_folder, "custom_elements")
custom_elements_folder
registry = PhotonRegistry(custom_elements_folder=custom_elements_folder)
registry.PHOTON_REGISTRIES
registry.activate()

# registry.list_available_elements()

registry.info("KMeans")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Preprocessing, OutputSettings
from photonai.optimization import FloatRange, Categorical, IntegerRange

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# DESIGN YOUR PIPELINE
settings = OutputSettings(project_folder="./tmp/")

my_pipe = Hyperpipe(
    "batching",
    optimizer="sk_opt",
    #                    optimizer_params={'n_configurations': 25},
    metrics=["ARI", "MI", "HCV", "FM"],
    best_config_metric="ARI",
    outer_cv=KFold(n_splits=2),
    inner_cv=KFold(n_splits=10),
    verbosity=1,
    output_settings=settings,
)


my_pipe += PipelineElement(
    "KMeans", hyperparameters={"n_clusters": IntegerRange(2, 12)}, random_state=777
)

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

debug = True
