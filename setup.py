from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# python setup.py build
# python setup.py build_ext --inplace


# This below is working and is fast. 
# But I cannot easily save the state
#         name="stepper.tdigest.exp_qtl",
#        sources=["stepper/tdigest/exp_qtl.pyx"],

extensions = [
    Extension(
        #name="stepper.incr_expanding_quantile",
        #sources=["stepper/incr_expanding_quantile.pyx"],


        #name="stepper.tdigest.tdigest",
        #sources=["stepper/tdigest/tdigest.pyx"],
        
        name="stepper.tdigest.exp_qtl2",
        sources=["stepper/tdigest/exp_qtl2.pyx",
                 "stepper/tdigest/tdigestloc.pyx"],
        
        language="c++",  # Use C++ since our T-Digest is C++
        include_dirs=[np.get_include(), 
                      "stepper/tdigest/tdigest_stubs.c",
                      ],
        # os.path.join(os.getcwd(), "path_to_tdigest_headers")
        # If your C++ T-Digest implementation is in separate files, add them here:
        # sources=["steppers/incr_expanding_quantile.pyx", "tdigest_stubs.c", "tdigest_cpp.cpp"],
    ),
        Extension(
        #name="stepper.incr_expanding_quantile",
        #sources=["stepper/incr_expanding_quantile.pyx"],


        #name="stepper.tdigest.tdigest",
        #sources=["stepper/tdigest/tdigest.pyx"],
        
        name="stepper.tdigest.tdigestloc",
        sources=["stepper/tdigest/tdigestloc.pyx"],
        
        language="c++",  # Use C++ since our T-Digest is C++
        include_dirs=[np.get_include(), 
                      "stepper/tdigest/tdigest_stubs.c",
                      ],
        # os.path.join(os.getcwd(), "path_to_tdigest_headers")
        # If your C++ T-Digest implementation is in separate files, add them here:
        # sources=["steppers/incr_expanding_quantile.pyx", "tdigest_stubs.c", "tdigest_cpp.cpp"],
    )
]

setup(
    name="YourProjectName",
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
)
