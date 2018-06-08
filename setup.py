from distutils.core import setup
from setuptools import find_packages

setup(
  name = 'photonai',
  packages = find_packages(),
      # 'photonai', 'photonai.base', 'photonai.configuration', 'photonai.documentation', 'photonai.examples',
      #         'photonai.genetics', 'photonai.helpers', 'photonai.investigator', 'photonai.modelwrapper', 'photonai.neuro',
      #         'photonai.optimization', 'photonai.photonlogger', 'photonai.sidepackages', 'photonai.test', 'photonai.validation'],
  version = '0.3.7',
  description = 'A Python-Based Hyperparameter optimization Toolbox for Neural Networks',
  author = 'PHOTON Team',
  author_email = 'hahnt@wwu.de',
  url = 'https://github.com/photonai-team/photonai.git', # use the URL to the github repo
  download_url = 'https://github.com/photonai-team/photonai/archive/0.3.7.tar.gz', # I'll explain this in a second
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
