from photonai.base import Hyperpipe
from sklearn.datasets import load_breast_cancer

X, _ = load_breast_cancer(True)

my_pipe = Hyperpipe.load_optimum_pipe("full_path/to/photon_best_model.photon")
predictions = my_pipe.predict(X)