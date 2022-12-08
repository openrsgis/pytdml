# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Boyi Shangguan, Kaixuan Wang
# Created: 2022-05-04
# Email: sgby@whu.edu.cn
#
# ------------------------------------------------------------------------------
#
# Copyright (c) 2022 OGC Training Data Markup Language for AI Standard Working Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------

"""Install pytdml."""

from setuptools import find_packages, setup
import io
import os.path


def read(filename, encoding='utf-8'):
    """read file contents"""
    full_path = os.path.join(os.path.dirname(__file__), filename)
    with io.open(full_path, encoding=encoding) as fh:
        contents = fh.read().strip()
    return contents


# get version number
with open(os.path.join(os.path.dirname(__file__), 'pytdml/__init__.py')) as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break

# use README.md for project long_description
long_description = read('README.md')

setup(
    name='pytdml',
    version=version,
    description='Parsing and encoding training datasets based on OGC Training Data Markup Language for AI ('
                'TrainingDML-AI) standard',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Boyi Shangguan',
    author_email='sgby@whu.edu.cn',
    url='https://github.com/TrainingDML/pytdml',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    tests_require=['pytest']
)
