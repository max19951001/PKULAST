# PKU LAnd Surface Temperature(PKULAST)

[![Build](https://github.com/tirzhu/PKULAST/actions/workflows/build.yml/badge.svg)](https://github.com/tirzhu/PKULAST/actions/workflows/build.yml)
[![PyPI package](https://badge.fury.io/py/pkulast.svg)](http://python.org/pypi/pkulast)

__pkulast__ is a Python module capable of retrieving land surface temperature (LST) from thermal infrared remote sensing data. It is built on-top of various scientific Python packages
([numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), etc.). With the help of __pkulast__, you can obtain algorithm coefficients of an existing LST retrieval algorithm for specific TIR sensors, or create a novel form of algorithm for new TIR sensor prototypes from scratch). 

- __Website:__ [PKULAST](http://github.com/tirzhu/pkulast)
- __Documentation:__ [PKULAST Documentation](http://readthedocs.com/pkulast)(TBD)


## Features

- __Native Python implementation.__ A native Python implementation for a variety of land surface temperature retrieval algorithms. To see the list of all supported algorithms, check this [link](http://readthedocs.com/pkulast).

- __Interface to MODTRAN.__ A MODTRAN wrapper class is implemented for reference purposes and integration. This provides access to atmospheric radiative transfer simulation &mdash;.
- __Conceptual Models.__ A conceptual forward model to simulate the top of atmosphere radiance; and inverse model to retrieve and validate land surface temperature.


## Dependencies

In most cases you will want to follow the requirements defined in the requirements/*.txt files in the package. 

### Base dependencies
```
scipy
numpy
satpy
metpy
pandas
pygrib
...
```

Note: Installing pygrib is complicated, please see: [pygrib install instructions using conda](https://anaconda.org/conda-forge/pygrib)

## Installation

To install pkulast, simply type the following command:

```bash
$ pip install pkulast.tar.gz
```

This will install the latest release from the Python package index.

## Configuration
__pkulast__ supports a configuration file in configparser syntax. The configuration is handled by the __pkulast.config__ module. The default file location is <install_dir>/pkulast.cfg but can be changed using the PKULAST_CONFIG environment variable.

## Contributing

This project is open for contributions. Here are some of the ways for
you to contribute:

- Bug reports/fix
- Features requests
- Use-case demonstrations
- Documentation updates

In case you want to implement PKULAST, please 
read our [Developer's Guide](http://readthedocs.com/pkulast) to help
you integrate your implementation in our API.

To make a contribution, just fork this repository, push the changes
in your fork, open up an issue, and make a Pull Request!

## Cite

If you used __pkulast__ in your research or project, please
cite [our work](https://doi.org/10.1109/JSTARS.2022.3217105):
```bibtex
@ARTICLE{2022pkulast,
   author = {Zhu, Jinshun and Ren, Huazhong and Ye, Xin and Teng, Yuanjian and Zeng, Hui and Liu, Yu and Fan, Wenjie},
   title = "{PKULAST-An Extendable Model for Land Surface Temperature Retrieval from Thermal Infrared Remote Sensing Data}",
   publisher = {IEEE Xplore},
   journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
   pages = {1--17},
   issn = {2151-1535},
   doi = {10.1109/JSTARS.2022.3217105},
   year = 2022,
   month = oct
}
```
