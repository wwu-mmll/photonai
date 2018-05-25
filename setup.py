from distutils.core import setup

setup(
  name = 'photon-ai',
  packages = ['photon-ai'], # this must be the same as the name above
  version = '0.2',
  description = 'A Python-Based Hyperparameter Optimization Toolbox for Neural Networks',
  author = 'PHOTON Team',
  author_email = 'hahnt@wwu.de',
  url = 'https://github.com/photon-ai-team/photon-ai.git', # use the URL to the github repo
  download_url = 'https://github.com/photon-ai-team/photon-ai/archive/0.2.tar.gz', # I'll explain this in a second
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
    'imblearn'
  ]
)