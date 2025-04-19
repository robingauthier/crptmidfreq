import pdb
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os
import glob

# python ./crptmidfreq/setup_stepperc.py build_ext --inplace

# Define the folder containing your .pyx files (adjust as needed)
pyx_folder = os.path.normpath(os.path.join(os.path.dirname(__file__), 'stepperc/'))

# Find all .pyx files in the folder
pyx_files = glob.glob(os.path.join(pyx_folder, "*.pyx"))

# Create a list of Extension objects for each .pyx file
extensions = []
for pyx_file in pyx_files:
    print(pyx_file)
    # Build all pyx files
    # Commented out to build all files: 
    # if not 'rolling_mean' in pyx_file:
    #     continue
    #print(pyx_file)
    # The module name is based on the file name (without extension)
    module_name = os.path.splitext(os.path.basename(pyx_file))[0]
    extensions.append(
        Extension(
            name='crptmidfreq.stepperc.{0}'.format(module_name),
            sources=[pyx_file],
            include_dirs=[numpy.get_include()],
            language="c++",  # necessary otherwise include ios issue
            extra_compile_args=["-O3", "-std=c++11"],  # to increase the speed of the code
        )
    )

setup(
    name="my_package",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        annotate=True  # Optional: generates an HTML annotation of the Cython code
    ),
)
