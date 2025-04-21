import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# python ./crptmidfreq/setup_hampel.py build_ext --inplace

# Create a list of Extension objects for each .pyx file
extensions = []
extensions.append(
    Extension(
        name='crptmidfreq.outlier.hampel_cython',
        sources=["crptmidfreq/outlier/hampel_cython.pyx"],
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
