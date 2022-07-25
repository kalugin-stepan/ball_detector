from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = '0.0.1'

ext_modules = [
    Pybind11Extension(
        'ball_detector',
        ['main.cpp'],
        include_dirs=['C:/Program Files/opencv/build/include'],
        library_dirs=['C:/Program Files/opencv/build/x64/vc15/lib'],
        libraries=['opencv_world460']
    )
]

setup(
    name='ball_detector',
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)