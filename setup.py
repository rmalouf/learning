## python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Naive Discriminative Learning',
  ext_modules = cythonize("_ndl.pyx"),
)