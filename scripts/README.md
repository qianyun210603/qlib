
- [Download Qlib Data](#Download-Qlib-Data)
  - [Download CN Data](#Download-CN-Data)
  - [Downlaod US Data](#Downlaod-US-Data)
  - [Download CN Simple Data](#Download-CN-Simple-Data)
  - [Help](#Help)
- [Using in Qlib](#Using-in-Qlib)
  - [US data](#US-data)
  - [CN data](#CN-data)


## Download Qlib Data


### Download CN Data

```bash
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### Downlaod US Data

> The US stock code contains 'PRN', and the directory cannot be created on Windows system: https://superuser.com/questions/613313/why-cant-we-make-con-prn-null-folder-in-windows

```bash
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

### Download CN Simple Data

```bash
python get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### Help

```bash
python get_data.py qlib_data --help
```

## Using in Qlib
> For more information: https://qlib.readthedocs.io/en/latest/start/initialization.html


### US data

```python
import qlib
from qlib.config import REG_US
provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_US)
```

### CN data

```python
import qlib
from qlib.config import REG_CN
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
```
