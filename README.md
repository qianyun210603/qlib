[![Python Versions](https://img.shields.io/pypi/pyversions/pyqlib.svg?logo=python&logoColor=white)](https://pypi.org/project/pyqlib/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey)](https://pypi.org/project/pyqlib/#files)
[![UnitTests](https://github.com/qianyun210603/qlib/actions/workflows/test_qlib_from_source.yml/badge.svg)](https://github.com/qianyun210603/qlib/actions/workflows/test_qlib_from_source.yml)
[![Documentation Status](https://readthedocs.org/projects/qlib/badge/?version=latest)](https://qlib.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/pyqlib)](LICENSE)


This is a fork of [Microsoft Qlib](http://github.com/microsoft/qlib) with some bug fixes and new features. Thanks for the Microsoft Qlib team for their great work.

# Motivation of this fork
I'm currently using qlib to resarch and execute some factor strategies on my own. However, some features I need are not avaliable in the original qlib also there are some bugs. So I forked the original qlib and added some features I need. I hope this fork can help some people who are also using qlib.
I will try to merge some non-debateable features/bugfix to the original qlib. But will keep some features that might be controversial in the fork only.

# Features
| Feature                        | Description                                                              | Status |
|--------------------------------|--------------------------------------------------------------------------| -- |
| Cross Sectional Factor         | Add cross-sectional factors such as cross-sectional ranking; average etc | Done |
| Orthogonalization preprocesser | Add preprocessers to do Schimit and Symetric Orthogonalization           | Done |
| Support non-adjusted data      | Add support for non-adjusted data                                        | Done |
| Enhanced plotting I            | Use rangebreak to allow Datetime axis in plottings                       | [Merged](https://github.com/microsoft/qlib/pull/1390) |
| Enhanced plotting II           | Add support for plotting factor returns                                  | Done | 
| Enhanced plotting III          | 1) Custom benchmark; 2) stratifying fix; 3) Colorbar enhancement         | [Merging](https://github.com/microsoft/qlib/pull/1413) | 
| Topk backtest engine I         | Allow sell in limit-up case and allow buy in limit-down case             | [Merged](https://github.com/microsoft/qlib/pull/1407) |
| Topk backtest engine II        | Sell names which are removed from instrument population (expired, delisted, removed from index etc)             | Done |
| Ops `Today`                    | Return Calendar days since BASE_DAY(1970-01-01)          | Done |
| Customize float data precision | Allow customize the precision of dumped data (to float64)          | Done |

<span style="font-size: xx-small; ">
<b>Note:</b>
<ul>
<li><b>Done:</b> The feature is implemented in this fork but will not be merged to the original qlib. Either because it is rejected by the original qlib team or because it is not suitable for the original qlib.</li>
<li><b>Merged:</b> The feature is merged to the original qlib.</li>
<li><b>Merging:</b> PR opened to the original qlib but not accepted yet.</li>
<li><b>Developing:</b> The feature is under development.</li>
</ul>
</span>


# Quick Start

This quick start guide tries to demonstrate
1. It's very easy to build a complete Quant research workflow and try your ideas with _Qlib_.
2. Though with *public data* and *simple models*, machine learning technologies **work very well** in practical Quant investment.

Here is a quick **[demo](https://terminalizer.com/view/3f24561a4470)** shows how to install ``Qlib``, and run LightGBM with ``qrun``.

## Installation

This table demonstrates the supported Python version of `Qlib`:

|               | install with pip | install from source  |       plot         |
| ------------- |:----------------:|:--------------------:|:------------------:|
| Python 3.7    |       :x:        | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.8    |       :x:        | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.9    |     :x:          | :heavy_check_mark:   | :heavy_check_mark: |

**Note**: 
1. **Conda** is suggested for managing your Python environment.
2. Please pay attention that installing cython in Python 3.6 will raise some error when installing ``Qlib`` from source. If users use Python 3.6 on their machines, it is recommended to *upgrade* Python to version 3.7 or use `conda`'s Python to install ``Qlib`` from source.
3. `Qlib`Requires `tables` package, `hdf5` in tables does not support python3.9.
4. This fork is not available on `pip`.

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

**Tips**: If you fail to install `Qlib` or run the examples in your environment,  comparing your steps and the [CI workflow](.github/workflows/test_qlib_from_source.yml) may help you find the problem.

### More details
Please refer the [readme](http://github.com/microsoft/qlib) and [documentation](https://qlib.readthedocs.io/en/latest) of the original qlib.


## Licence
The forked version inherits the licence of the original qlib. See [LICENSE](LICENSE) for details.
