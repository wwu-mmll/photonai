try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages


__version__ = '0.5.0'

setup(
  name='photonai',
  packages=find_packages(),
  include_package_data=True,
  version=__version__,
  description='A Python-Based Hyperparameter optimization Toolbox for Neural Networks',
  author='PHOTON Team',
  author_email='hahnt@wwu.de',
  url='https://github.com/photonai-team/photonai.git',
  download_url='https://github.com/photonai-team/photonai/archive/' + __version__ + '.tar.gz',
  keywords=['machine learning', 'deep learning', 'neural networks', 'hyperparameter'],
  classifiers=[],
  install_requires=[
        'numpy',
        'matplotlib',
        'progressbar2',
        'Pillow',
        'scikit-learn',
        'keras',
        'nilearn==0.5.0',
        'pandas',
        'nibabel',
        'pandas',
        'six',
        'h5py',
        'xlrd',
        'plotly',
        'imblearn',
        'pymodm==0.4.1',
        'scipy',
        'statsmodels',
        'flask',
        'prettytable',
        'scikit-optimize',
        'scikit-image',
        'seaborn',
        'joblib',
      'fasteners',
      'dask',
      'distributed'
  ]
)
