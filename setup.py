from setuptools import setup


long_description_string = "For detailed information\n" \
                          "Please look at : " \
                          "https://github.com/AyberkYavuz/artificial_neural_network_automation"

setup(
    # Name of the package
    name='artificial_neural_network_model_automation',

    # Packages to include into the distribution
    packages=['artificial_neural_network_model_automation', 'helper', 'tests'],

    # Start with a small number and increase it with every change you make
    # https://semver.org
    version='0.0.1',

    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    # For example: MIT
    license='MIT',

    # Short description of your library
    description='This repository is for automating artificial neural network model creation with tabular data using Keras framework.',

    # Long description of your library
    long_description="""{}""".format(long_description_string),
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