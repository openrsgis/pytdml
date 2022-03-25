# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------

"""Install pytdml."""

from setuptools import find_packages, setup
import os
import os.path

# don't install dependencies when building win readthedocs
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# get version number
with open(os.path.join(os.path.dirname(__file__), 'pytdml/__init__.py')) as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break

# use README.md for project long_description
with open('README.md') as f:
    readme = f.read()

setup(
    name='pytdml',
    version=version,
    description='Parsing and encoding training datasets based on TrainingDML-AI',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Boyi Shangguan',
    author_email='sgby@whu.edu.cn',
    url='https://github.com/sgby/pytdml',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "dataclasses;python_version<'3.7'",
        "lxml",
    ] if not on_rtd else [],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    tests_require=['pytest']
)