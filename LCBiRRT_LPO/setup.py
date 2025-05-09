# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from setuptools import find_packages
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
# setup_args = generate_distutils_setup(packages=['srmt', 'srmt.*'])
    
# setup_args = generate_distutils_setup(
#     packages=['srmt'])
#
# setup(**setup_args)
# LCBiRRT_LPO

setup(
    name='lcbirrt_lpo',
    version='1.0.1',
    description='Local path optimization in the latent space using learned distance gradient',
    packages=find_packages(include=['lcbirrt_lpo', 'lcbirrt_lpo.*']),
)
