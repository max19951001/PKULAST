# PKU LAnd Surface Temperature(PKULAST)

[![Github Actions Status](https://github.com/hexlet-boilerplates/python-package/workflows/Python%20CI/badge.svg)](https://github.com/hexlet-boilerplates/python-package/actions)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)
[![Maintainability](https://api.codeclimate.com/v1/badges/df66c0cbbeca7d822f23/maintainability)](https://codeclimate.com/github/hexlet-boilerplates/python-package/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/df66c0cbbeca7d822f23/test_coverage)](https://codeclimate.com/github/hexlet-boilerplates/python-package/test_coverage)



__pkulast__ is a Python module capable of retrieving land surface temperature from thermal infrared remote sensing data. It is built on-top of various scientific Python packages
([numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/)).

- __Website:__ [PKULAST](http://github.com/tirzhu/pkulast)
- __Documentation:__ [PKULAST Documentation](http://readthedocs.com/pkulast)(TBD)


## Features

- __Native Python implementation.__ A native Python implementation for a variety of land surface temperature retrieval algorithms. To see the list of all supported algorithms, check this [link](http://scikit.ml/#classifiers).

- __Interface to MODTRAN.__ A MODTRAN wrapper class is implemented for reference purposes and integration. This provides access to atmospheric radiative transfer simulation &mdash; the reference standard in the field.
- __Conceptual Models.__ A conceptual forward model to simulate the top of atmosphere radiance; and inverse model to retrieve and validate land surface temperature.


## Dependencies

In most cases you will want to follow the requirements defined in the requirements/*.txt files in the package. 

### Base dependencies
```
scipy
numpy
...
```

Note: Installing pygrib is complicated, please see: [pygrib install instructions](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions)

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
cite [our work](https://doi.org/xxxx.xxxx):

```bibtex
@ARTICLE{2022pkulast,
   author = {{Zhu}, J.},
   title = "{PKU LAnd Surface Temperature(PKULAST)}",
   publisher = {GitHub},
   journal = {GitHub repository},
   year = 2022,
   month = feb
}
```
