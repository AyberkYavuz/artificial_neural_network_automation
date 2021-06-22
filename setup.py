from distutils.core import setup
from setuptools import find_packages
import os


# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
    # Name of the package
    name='artificial_neural_network_model_automation',

    # Packages to include into the distribution
    packages=find_packages('.'),

    # Start with a small number and increase it with every change you make
    # https://semver.org
    version='0.0.1',

    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    # For example: MIT
    license='MIT',

    # Short description of your library
    description='This repository is for automating artificial neural network '
                'model creation with tabular data using Keras framework.',

    # Long description of your library
    long_description=long_description,
    long_description_context_type='text/markdown',

    # Your name
    author='Ayberk Yavuz',

    # Your email
    author_email='ayberk.yavuz06@gmail.com',

    # Either the link to your github or to your website
    url='https://github.com/AyberkYavuz/artificial_neural_network_automation',

    # Link from which the project can be downloaded
    download_url='https://github.com/AyberkYavuz/artificial_neural_network_automation/archive/refs/tags/v0.0.1.tar.gz',

    # List of keyword arguments
    keywords=['Keras', 'artificial neural networks', 'automation',
              'supervised learning', 'tabular data'],

    # List of packages to install with this one
    install_requires=['tensorflow>=2.4.1', 'Keras>=2.4.3', 'pandas>=1.2.3',
                      'numpy>=1.19.5', 'sklearn>=0.0', 'joblib>=0.16.0',
                      'pydot>=1.4.2'],

    # https://pypi.org/classifiers/
    classifiers=['Programming Language :: Python :: 3', 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'License :: OSI Approved :: MIT License'],

    python_requires=">=3.8"
)