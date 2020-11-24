import os
from setuptools import find_packages
from setuptools import setup
from hstest.constants import package_name, __version__

core_branch = os.getenv("CORE_BRANCH", "master")
# core_branch = os.getenv("CORE_BRANCH",'CORE-22-add-augmentation-to-image-classi')
REQUIRED_PACKAGES = ['brevetti-ai @ git+https://bitbucket.org/criterionai/core@' + core_branch]

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
