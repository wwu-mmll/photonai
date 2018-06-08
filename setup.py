from distutils.core import setup
from setuptools import find_packages

setup(
  name = 'photonai',
  packages = find_packages(),
  include_package_data=True,
  version = '0.1.2',
  description = 'A Python-Based Hyperparameter optimization Toolbox for Neural Networks',
  author = 'PHOTON Team',
  author_email = 'hahnt@wwu.de',
  url = 'https://github.com/photonai-team/photonai.git', # use the URL to the github repo
  download_url = 'https://github.com/photonai-team/photonai/archive/0.1.2.tar.gz', # I'll explain this in a second
  keywords = ['machine learning', 'deep learning', 'neural networks', 'hyperparameter'], # arbitrary keywords
  classifiers = [],
  install_requires = [
        'numpy',
        'matplotlib',
        'tensorflow',
        'slackclient',
        'progressbar2',
        'Pillow',
        'scikit-learn',
        'keras',
        'nilearn',
        'pandas',
        'nibabel',
        'pandas',
        'six',
        'h5py',
        'xlrd',
        'plotly',
        'imblearn',
        'pymodm',
        'scipy',
        'statsmodels',
        'flask'
  ]
)
