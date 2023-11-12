from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['ts_plant_probe'],
    package_dir={'': 'src'}
)

setup(**d)