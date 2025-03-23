from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
# python setup.py build_ext --inplace 
# python ./crptmidfreq/setup.py build_ext --inplace --force
# it will copy from the build folder that is in the current folder
# Cmd +  Shift + V to preview the html file
# change the .pyx to trigger annotation

extensions = [
    Extension(
        name="crptmidfreq.stepper.tdigest.exp_qtl2",
        sources=["crptmidfreq/stepper/tdigest/exp_qtl2.pyx",
                 ],        # "stepper/tdigest/tdigestloc.pyx"
        language="c++",  # Use C++ since our T-Digest is C++
        include_dirs=[np.get_include(), 
                      #"stepper/tdigest/tdigest_stubs.c",
                      "crptmidfreq/stepper/tdigest/",
                      ],
    ),
    Extension(
        name="crptmidfreq.stepper.tdigest.tdigestloc",
       sources=["crptmidfreq/stepper/tdigest/tdigestloc.pyx"],
        language="c++",  # Use C++ since our T-Digest is C++
        include_dirs=[np.get_include(), 
                      #"stepper/tdigest/tdigest_stubs.c",
                      "crptmidfreq/stepper/tdigest/",
                      ],
    ),
    
    Extension(
        name="crptmidfreq.stepper.p2_algo.p2_quantile",
       sources=["crptmidfreq/stepper/p2_algo/p2_quantile.pyx",
                "crptmidfreq/stepper/p2_algo/p2.cpp"],
        language="c++",  # Use C++ since our T-Digest is C++
        include_dirs=[np.get_include(), 
                      "crptmidfreq/stepper/p2_algo/",
                      ],
    ),
        
    Extension(
        name="crptmidfreq.stepper.p2_algo.exp_qtl2",
       sources=["crptmidfreq/stepper/p2_algo/exp_qtl2.pyx"],
        language="c++",  # Use C++ since our T-Digest is C++
        include_dirs=[np.get_include(), 
                      "crptmidfreq/stepper/p2_algo/",
                      ],
    )

]


setup(
    name="YourProjectName",
    ext_modules=cythonize(extensions, 
        annotate=True,
        compiler_directives={
            "linetrace": False,     # Only if profiling
            "language_level": 3,
        }),
    zip_safe=False,
)
