try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

# from setuptools.command.install import install
#
# class Modulenstalltio


__version__ = '2.1.0'

setup(
    name='photonai',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    description="""
PHOTONAI
is a rapid prototyping framework enabling (not so experienced) users to build, train, optimize, evaluate,
and share even complex machine learning (ML) pipelines with very high efficiency.
""",
    author='PHOTONAI Team',
    author_email='hahnt@wwu.de',
    url='https://github.com/mmll-wwu/photonai.git',
    download_url='https://github.com/wwu-mmll/photonai/archive/' + __version__ + '.tar.gz',
    keywords=['machine learning', 'deep learning', 'neural networks', 'hyperparameter'],
    classifiers=[],
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'keras',
        'pandas',
        'plotly',
        'imbalanced-learn',
        'pymodm',
        'scipy',
        'statsmodels',
        'prettytable',
        'seaborn',
        'joblib',
        'dask==2.30.0',
        'distributed==2.30.1',
        'scikit-optimize',
        'xlrd']
)
