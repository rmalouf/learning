## python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
  name = 'Naive Discriminative Learning',
  ext_modules = cythonize("_ndl.pyx"),
  include_dirs = [np.get_include()]
)