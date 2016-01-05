from setuptools import setup
from setuptools import find_packages


setup(name='GmdhPy',
      version='0.1.1',
      description='Multilayered group method of data handling of Machine learning for Python',
      author='Konstantin Kolokolov',
      author_email='konstantin.kolokolov@gmail.com',
      url='https://github.com/kvoyager/GmdhPy',
      download_url='https://github.com/kvoyager/GmdhPy/archive/master.zip',
      license='MIT',
      install_requires=['numpy', 'six', 'sklearn', 'matplotlib', 'multiprocessing'],
      extras_require={
          'graphviz': ['graphviz'],
      },
      packages=find_packages()
      )
