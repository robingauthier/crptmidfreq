from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# python setup.py build_ext --inplace


extensions = [
    Extension(
        name="stepper.tdigest.exp_qtl2",
        sources=["stepper/tdigest/exp_qtl2.pyx",
                 "stepper/tdigest/tdigestloc.pyx"],        
        language="c++",  # Use C++ since our T-Digest is C++
        include_dirs=[np.get_include(), 
                      "stepper/tdigest/tdigest_stubs.c",
                      ],
    ),
        Extension(
        name="stepper.tdigest.tdigestloc",
        sources=["stepper/tdigest/tdigestloc.pyx"],
        language="c++",  # Use C++ since our T-Digest is C++
        include_dirs=[np.get_include(), 
                      "stepper/tdigest/tdigest_stubs.c",
                      ],
    )
]

setup(
    name="YourProjectName",
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
)
