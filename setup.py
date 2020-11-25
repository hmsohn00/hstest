import os
from setuptools import find_packages
from setuptools import setup
from hstest.constants import package_name, __version__

core_branch = os.getenv("CORE_BRANCH", "master")
REQUIRED_PACKAGES = ['brevetti-ai @ git+https://git@bitbucket.org/criterionai/core.git@' + core_branch]

setup(
    name=package_name,
    version=__version__,
    url='https://github.com/hmsohn00/hstest.git',
    author='Hyemee Sohn',
    author_email='hmsohn00@gmail.com',
    packages=find_packages(),
    description='Test model type by Hyemee',
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES
)
