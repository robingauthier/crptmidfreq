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
    # The module name is based on the file name (without extension)
    module_name = os.path.splitext(os.path.basename(pyx_file))[0]
    extensions.append(
        Extension(
            name=f'crptmidfreq.stepperc.{module_name}',
            sources=[pyx_file],
            include_dirs=[numpy.get_include()],
            language="c++",  # necessary otherwise include ios issue
        )
    )
#        name="crptmidfreq.stepper.tdigest.tdigestloc",
#       sources=["crptmidfreq/stepper/tdigest/tdigestloc.pyx"],
# print(extensions)
#pdb.set_trace()
setup(
    name="my_package",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
        annotate=True  # Optional: generates an HTML annotation of the Cython code
    ),
)
