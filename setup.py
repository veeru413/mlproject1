from setuptools import setup, find_packages # find packages is used to automatically find all packages in the project
from typing import List
# setup function is used to specify the details of the package

#just meta data of the package, we can add more details like author, description, etc.

# sometimes we might have 100s of packages so we use a function here
def get_requirements(filename: str) -> List[str]:
    requirements = []
    with open(filename) as f:
        requirements = [req.strip() for req in f if req.strip()]
        # ignore editable installs and local package paths
        requirements = [req for req in requirements if not req.startswith('-e')]
    return requirements

setup(
    name='mlproject', # name of the package
    version='0.0.1', # version of the package 
    author='veeru',
    author_email='veerendrapatil2123@gmail.com',
    packages=find_packages(), # find all packages in the project
    #install_requires=['pandas','numpy','seaborn'] # list of dependencies required to run the package
    install_requires=get_requirements('requirements.txt') # we can also use a function to get the requirements from a file like requirements.txt
)
# -e . used in requirements.txt so that when we run the requirements.txt file by using pip install -r requirements.txt command, it will install the package in editable mode, 
# which means that any changes made to the package will be reflected in the installed package without having to reinstall it. This is useful during development when we want to test changes to the package without having to reinstall it every time. When we run pip install -r requirements.txt command, it will read the requirements.txt file and install all the dependencies mentioned in it, including the package itself if -e . is included. 
# it will also run setup.py file and install the package as well as the dependencies mentioned in the 
# requirements.txt file
'''
How packages=find_packages() works
find_packages() scans the project directories and finds Python packages automatically.

A Python package is usually a folder that contains an __init__.py file. For example:

mlproject/
__init__.py
data_load.py
model.py
If that exists, find_packages() can include mlproject as an installable package.

What __init__.py means
__init__.py is what tells Python: “this directory is a package.”

It allows you to do imports like:

from mlproject import data_load
import mlproject
Without __init__.py, older Python versions did not treat the folder as a package. In modern Python it is still best practice to include it for package behavior and explicitness.

__init__.py can be empty, or it can also:

expose public functions / classes
define package-level imports
set __version__, etc.
What src/ means
src/ is a common project layout, but it is optional.

Example layout:

setup.py
src/
mlproject/
__init__.py
train.py
predict.py
'''