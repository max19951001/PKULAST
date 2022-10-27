# -*- coding:utf-8 -*-
# Copyright (c) 2021-2022.

################################################################
# The contents of this file are subject to the GPLv3 License
# you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# https://www.gnu.org/licenses/gpl-3.0.en.html

# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations
# under the License.

# The Original Code is part of the PKULAST python package.

# Initial Dev of the Original Code is Jinshun Zhu, PhD Student,
# Institute of Remote Sensing and Geographic Information System,
# Peking Universiy Copyright (C) 2022
# All Rights Reserved.

# Contributor(s): Jinshun Zhu (created, refactored and updated original code).
###############################################################


"""Setup file for pkulast.
"""
import os.path
from setuptools import setup, find_packages

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep))

NAME = 'pkulast'
with open('README.md', 'r') as readme:
    README = readme.read()


try:
    import setuptools_scm.integration
    setuptools_scm.integration.find_files = lambda _:[]
except ImportError:
    pass


requires = []
# 'numpy>=1.4.1'
# 'pint'>='0.8',
# 'matplotlib>=3.4.1',
# 'pyproj>=3.0.1',
# 'pysolar>=0.9',
# 'GDAL>=3.2.2',
# 'rasterio>=1.2.1',
# 'logging>=0.4.9',
# 'appdirs>=1.4.4',
# 'PyYAML>=5.4.1',
# 'scipy>=1.3.1',
# 'beautifulsoup4>=4.9.3',
# 'h5py>=3.2.1',
# 'pyshp>=2.1.3',
# 'requests>=2.25.1',
# 'pygrib>=2.1.3',
# 'pyresample>=1.18.1',
# 'satpy>=0.27.0',
# 'coloredlogs>=15.0',
# 'sklearn',
# 'polar2grid>=2.4.1',
# 'xarray']

test_requires = []

extras_require = {
    'doc': ['sphinx'],
}
all_extras = []
for extra_deps in extras_require.values():
    all_extras.extend(extra_deps)
extras_require['all'] = list(set(all_extras))

setup(
    name='pkulast',
    version='0.1.6',
    keywords='modtran infrared',
    description='PKU LAnd Surface Temperature(PKULAST)',
    long_description=README,
    long_description_content_type="text/markdown",
    license='LGPL License',
    url='https://github.com/tirzhu/pkulast_stable',
    author='Jinshun Zhu',
    author_email='tirzhu@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['*.yml', '*.yaml'],
    },
    platforms='any',
    zip_safe=False,
    install_requires=requires,
    tests_require=test_requires,
    python_requires='>=3.7',
    extras_require=extras_require,
    )
