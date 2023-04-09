[![Python Versions](https://img.shields.io/pypi/pyversions/pyqlib.svg?logo=python&logoColor=white)](https://pypi.org/project/pyqlib/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey)](https://pypi.org/project/pyqlib/#files)
[![Github Actions Test Status](https://github.com/qianyun210603/qlib/actions/workflows/Tests.yml/badge.svg)](https://github.com/qianyun210603/qlib/actions)
[![Documentation Status](https://readthedocs.org/projects/qlib/badge/?version=latest)](https://qlib.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/pyqlib)](LICENSE)


Qlib is an AI-oriented quantitative investment platform, which aims to realize the potential, empower the research, and create the value of AI technologies in quantitative investment.

It contains the full ML pipeline of data processing, model training, back-testing; and covers the entire chain of quantitative investment: alpha seeking, risk modeling, portfolio optimization, and order execution. 

With Qlib, users can easily try ideas to create better Quant investment strategies.

For more details, please refer to our paper ["Qlib: An AI-oriented Quantitative Investment Platform"](https://arxiv.org/abs/2009.11189).

# Quick Start

This quick start guide tries to demonstrate
1. It's very easy to build a complete Quant research workflow and try your ideas with _Qlib_.
2. Though with *public data* and *simple models*, machine learning technologies **work very well** in practical Quant investment.

Here is a quick **[demo](https://terminalizer.com/view/3f24561a4470)** shows how to install ``Qlib``, and run LightGBM with ``qrun``. **But**, please make sure you have already prepared the data following the [instruction](#data-preparation).


## Installation

This table demonstrates the supported Python version of `Qlib`:
|               | install with pip           | install from source  | plot |
| ------------- |:---------------------:|:--------------------:|:----:|
| Python 3.7    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.8    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.9    | :x:                   | :heavy_check_mark:   | :heavy_check_mark: |

**Note**: 
1. **Conda** is suggested for managing your Python environment.
1. Please pay attention that installing cython in Python 3.6 will raise some error when installing ``Qlib`` from source. If users use Python 3.6 on their machines, it is recommended to *upgrade* Python to version 3.7 or use `conda`'s Python to install ``Qlib`` from source.
1. For Python 3.9, `Qlib` supports running workflows such as training models, doing backtest and plot most of the related figures (those included in [notebook](examples/workflow_by_code.ipynb)). However, plotting for the *model performance* is not supported for now and we will fix this when the dependent packages are upgraded in the future.
1. `Qlib`Requires `tables` package, `hdf5` in tables does not support python3.9. 

### Install from package manager
Current the fork is not available in pip or conda

### Install from source
Users can install the latest forked version ``Qlib`` by the source code according to the following steps:

* Before installing ``Qlib`` from source, users need to install some dependencies:

  ```bash
  pip install numpy
  pip install --upgrade  cython
  ```

* Clone the repository and install ``Qlib`` as follows.
    ```bash
    git clone https://github.com/qianyun210603/qlib.git && cd qlib
    pip install .
    ```
  **Note**:  You can install Qlib with `python setup.py install` as well. But it is not the recommended approach. It will skip `pip` and cause obscure problems. For example, **only** the command ``pip install .`` **can** overwrite the stable version installed by ``pip install pyqlib``, while the command ``python setup.py install`` **can't**.

**Tips**: If you fail to install `Qlib` or run the examples in your environment,  comparing your steps and the [CI workflow](.github/workflows/Tests.yml) may help you find the problem.

## Data Preparation
TODO


## Licence
TODO
