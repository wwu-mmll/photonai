from sklearn.datasets import load_diabetes
from photonai import RegressionPipe

my_pipe = RegressionPipe('diabetes')
my_pipe.set_default_pipeline()
# load data and train
X, y = load_diabetes(return_X_y=True)
my_pipe.fit(X, y)
