from setuptools import setup
from setuptools import find_packages


setup(name='GmdhPy',
      version='2.0',
      description='Self-organizing deep learning polynomial neural network for Python (Multilayered group method of data handling)',
      author='kvoyager',
      author_email='konstantin.kolokolov@gmail.com',
      url='https://github.com/kvoyager/GmdhPy',
      download_url='https://github.com/kvoyager/GmdhPy/archive/master.zip',
      license='MIT',
      install_requires=['numpy', 'six', 'scikit-learn', 'pandas'],
      extras_require={
          'graphviz': ['graphviz', 'matplotlib'],
      },
      packages=find_packages()
      )
