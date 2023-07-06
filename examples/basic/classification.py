from photonai import ClassificationPipe
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

my_pipe = ClassificationPipe('breast_cancer_analysis')
my_pipe.set_default_pipeline()
my_pipe.fit(X, y)
