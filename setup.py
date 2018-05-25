from distutils.core import setup

setup(
  name = 'photon-ai',
  packages = ['photon-ai'], # this must be the same as the name above
  version = 'alpha0.2',
  description = 'A Python-Based Hyperparameter Optimization Toolbox for Neural Networks',
  author = 'PHOTON Team',
  author_email = 'hahnt@wwu.de',
  url = 'https://github.com/photon-team/photon.git', # use the URL to the github repo
  download_url = 'https://github.com/photon-team/photon/archive/alpha0.2.tar.gz', # I'll explain this in a second
  keywords = ['machine learning', 'deep learning', 'neural networks', 'hyperparameter'], # arbitrary keywords
  classifiers = [],
)