from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
  name = 'Gaussfit function',
  ext_modules = cythonize("gaussfit.pyx", language_level = "3"),
  #cmdclass = {'build_ext': build_ext},
  include_dirs = [numpy.get_include()] #Include directory not hard-wired
  )

