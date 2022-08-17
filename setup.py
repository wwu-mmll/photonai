try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages


__version__ = '2.2.1'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='photonai',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    description="PHOTONAI is a high level python API for designing and optimizing machine learning pipelines.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "GPLv3",
    author='PHOTONAI Team',
    author_email='hahnt@wwu.de',
    url='https://www.photon-ai.com/',
    project_urls={
        "Source Code": "https://github.com/wwu-mmll/photonai/",
        "Documentation": "https://wwu-mmll.github.io/photonai/",
        "Bug Tracker": "https://github.com/wwu-mmll/photonai/issues",
    },
    download_url='https://pypi.org/project/photonai/#files',
    keywords=['machine learning', 'deep learning', 'neural networks', 'hyperparameter'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pandas',
        'plotly',
        'imbalanced-learn',
        'pymodm',
        'scipy',
        'statsmodels',
        'prettytable',
        'seaborn',
        'joblib',
        'dask>=2021.10.0',
        'distributed',
        'scikit-optimize',
        'xlrd']
)
