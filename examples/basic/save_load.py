from photonai.base import Hyperpipe
from sklearn.datasets import load_breast_cancer

X, _ = load_breast_cancer(True)

# After optimization is finished, PHOTONAI saves the user's pipeline
# fitted with the best hyperparameter configuration found
# as "photon_best_model.photon" in the project's result folder.
# this is done automatically, however the use may do so manually by calling
# my_pipe.save_optimum_pipe('/home/photon_user/photon_test/optimum_pipe.photon')

my_pipe = Hyperpipe.load_optimum_pipe("full_path/to/photon_best_model.photon")
predictions = my_pipe.predict(X)
