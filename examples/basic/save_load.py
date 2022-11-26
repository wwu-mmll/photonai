from photonai import Hyperpipe
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(True)

# After optimization is finished, PHOTONAI saves the user's pipeline
# fitted with the best hyperparameter configuration found
# as "best_model.photonai" in the project's result folder.
# this is done automatically, however the use may do so manually by calling
# my_pipe.save_optimum_pipe('/home/photon_user/photon_test/optimum_pipe.photon')

my_pipe = Hyperpipe.load_optimum_pipe("full_path/to/best_model.photonai")
predictions = my_pipe.predict(X)

# get permutation importances posthoc
reloaded_hyperpipe = Hyperpipe.reload_hyperpipe("full_path/to/results_folder/", X, y)
post_hoc_perm_importances = Hyperpipe.get_permutation_feature_importances(n_repeats=5, random_state=0)
