# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import re
import copy
import json
import yaml
import redis
import bisect
import shutil
import difflib
import hashlib
import datetime
import requests
import tempfile
import importlib
import contextlib
import collections
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Text, Optional

from ..config import C
from ..log import get_module_logger, set_log_with_config

log = get_module_logger("utils")


#################### Server ####################
def get_redis_connection():
    """get redis connection instance."""
    return redis.StrictRedis(host=C.redis_host, port=C.redis_port, db=C.redis_task_db)


#################### Data ####################
def read_bin(file_path, start_index, end_index):
    with open(file_path, "rb") as f:
        # read start_index
        ref_start_index = int(np.frombuffer(f.read(4), dtype="<f")[0])
        si = max(ref_start_index, start_index)
        if si > end_index:
            return pd.Series(dtype=np.float32)
        # calculate offset
        f.seek(4 * (si - ref_start_index) + 4)
        # read nbytes
        count = end_index - si + 1
        data = np.frombuffer(f.read(4 * count), dtype="<f")
        series = pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
    return series


def np_ffill(arr: np.array):
    """
    forward fill a 1D numpy array

    Parameters
    ----------
    arr : np.array
        Input numpy 1D array
    """
    mask = np.isnan(arr.astype(float))  # np.isnan only works on np.float
    # get fill index
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


#################### Search ####################
def lower_bound(data, val, level=0):
    """multi fields list lower bound.

    for single field list use `bisect.bisect_left` instead
    """
    left = 0
    right = len(data)
    while left < right:
        mid = (left + right) // 2
        if val <= data[mid][level]:
            right = mid
        else:
            left = mid + 1
    return left


def upper_bound(data, val, level=0):
    """multi fields list upper bound.

    for single field list use `bisect.bisect_right` instead
    """
    left = 0
    right = len(data)
    while left < right:
        mid = (left + right) // 2
        if val >= data[mid][level]:
            left = mid + 1
        else:
            right = mid
    return left


#################### HTTP ####################
def requests_with_retry(url, retry=5, **kwargs):
    while retry > 0:
        retry -= 1
        try:
            res = requests.get(url, timeout=1, **kwargs)
            assert res.status_code in {200, 206}
            return res
        except AssertionError:
            continue
        except Exception as e:
            log.warning("exception encountered {}".format(e))
            continue
    raise Exception("ERROR: requests failed!")


#################### Parse ####################
def parse_config(config):
    # Check whether need parse, all object except str do not need to be parsed
    if not isinstance(config, str):
        return config
    # Check whether config is file
    if os.path.exists(config):
        with open(config, "r") as f:
            return yaml.safe_load(f)
    # Check whether the str can be parsed
    try:
        return yaml.safe_load(config)
    except BaseException:
        raise ValueError("cannot parse config!")


#################### Other ####################
def drop_nan_by_y_index(x, y, weight=None):
    # x, y, weight: DataFrame
    # Find index of rows which do not contain Nan in all columns from y.
    mask = ~y.isna().any(axis=1)
    # Get related rows from x, y, weight.
    x = x[mask]
    y = y[mask]
    if weight is not None:
        weight = weight[mask]
    return x, y, weight


def hash_args(*args):
    # json.dumps will keep the dict keys always sorted.
    string = json.dumps(args, sort_keys=True, default=str)  # frozenset
    return hashlib.md5(string.encode()).hexdigest()


def parse_field(field):
    # Following patterns will be matched:
    # - $close -> Feature("close")
    # - $close5 -> Feature("close5")
    # - $open+$close -> Feature("open")+Feature("close")
    if not isinstance(field, str):
        field = str(field)
    return re.sub(r"\$(\w+)", r'Feature("\1")', re.sub(r"(\w+\s*)\(", r"Operators.\1(", field))


def get_module_by_module_path(module_path):
    """Load module path

    :param module_path:
    :return:
    """

    if module_path.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location("", module_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    return module


def get_cls_kwargs(config: Union[dict, str], module) -> (type, dict):
    """
    extract class and kwargs from config info

    Parameters
    ----------
    config : [dict, str]
        similar to config

    module : Python module
        It should be a python module to load the class type

    Returns
    -------
    (type, dict):
        the class object and it's arguments.
    """
    if isinstance(config, dict):
        # raise AttributeError
        klass = getattr(module, config["class"])
        kwargs = config.get("kwargs", {})
    elif isinstance(config, str):
        klass = getattr(module, config)
        kwargs = {}
    else:
        raise NotImplementedError(f"This type of input is not supported")
    return klass, kwargs


def init_instance_by_config(
    config: Union[str, dict, object], module=None, accept_types: Union[type, Tuple[type]] = (), **kwargs
) -> object:
    """
    get initialized instance with config

    Parameters
    ----------
    config : Union[str, dict, object]
        dict example.
            {
                'class': 'ClassName',
                'kwargs': dict, #  It is optional. {} will be used if not given
                'model_path': path, # It is optional if module is given
            }
        str example.
            "ClassName":  getattr(module, config)() will be used.
        object example:
            instance of accept_types
    module : Python module
        Optional. It should be a python module.
        NOTE: the "module_path" will be override by `module` arguments

    accept_types: Union[type, Tuple[type]]
        Optional. If the config is a instance of specific type, return the config directly.
        This will be passed into the second parameter of isinstance.

    Returns
    -------
    object:
        An initialized object based on the config info
    """
    if isinstance(config, accept_types):
        return config

    if module is None:
        module = get_module_by_module_path(config["module_path"])

    klass, cls_kwargs = get_cls_kwargs(config, module)
    return klass(**cls_kwargs, **kwargs)


def compare_dict_value(src_data: dict, dst_data: dict):
    """Compare dict value

    :param src_data:
    :param dst_data:
    :return:
    """

    class DateEncoder(json.JSONEncoder):
        # FIXME: This class can only be accurate to the day. If it is a minute,
        # there may be a bug
        def default(self, o):
            if isinstance(o, (datetime.datetime, datetime.date)):
                return o.strftime("%Y-%m-%d %H:%M:%S")
            return json.JSONEncoder.default(self, o)

    src_data = json.dumps(src_data, indent=4, sort_keys=True, cls=DateEncoder)
    dst_data = json.dumps(dst_data, indent=4, sort_keys=True, cls=DateEncoder)
    diff = difflib.ndiff(src_data, dst_data)
    changes = [line for line in diff if line.startswith("+ ") or line.startswith("- ")]
    return changes


def get_or_create_path(path: Optional[Text] = None, return_dir: bool = False):
    """Create or get a file or directory given the path and return_dir.

    Parameters
    ----------
    path: a string indicates the path or None indicates creating a temporary path.
    return_dir: if True, create and return a directory; otherwise c&r a file.

    """
    if path:
        if return_dir and not os.path.exists(path):
            os.makedirs(path)
        elif not return_dir:  # return a file, thus we need to create its parent directory
            xpath = os.path.abspath(os.path.join(path, ".."))
            if not os.path.exists(xpath):
                os.makedirs(xpath)
    else:
        temp_dir = os.path.expanduser("~/tmp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if return_dir:
            _, path = tempfile.mkdtemp(dir=temp_dir)
        else:
            _, path = tempfile.mkstemp(dir=temp_dir)
    return path


@contextlib.contextmanager
def save_multiple_parts_file(filename, format="gztar"):
    """Save multiple parts file

    Implementation process:
        1. get the absolute path to 'filename'
        2. create a 'filename' directory
        3. user does something with file_path('filename/')
        4. remove 'filename' directory
        5. make_archive 'filename' directory, and rename 'archive file' to filename

    :param filename: result model path
    :param format: archive format: one of "zip", "tar", "gztar", "bztar", or "xztar"
    :return: real model path

    Usage::

        >>> # The following code will create an archive file('~/tmp/test_file') containing 'test_doc_i'(i is 0-10) files.
        >>> with save_multiple_parts_file('~/tmp/test_file') as filename_dir:
        ...   for i in range(10):
        ...       temp_path = os.path.join(filename_dir, 'test_doc_{}'.format(str(i)))
        ...       with open(temp_path) as fp:
        ...           fp.write(str(i))
        ...

    """

    if filename.startswith("~"):
        filename = os.path.expanduser(filename)

    file_path = os.path.abspath(filename)

    # Create model dir
    if os.path.exists(file_path):
        raise FileExistsError("ERROR: file exists: {}, cannot be create the directory.".format(file_path))

    os.makedirs(file_path)

    # return model dir
    yield file_path

    # filename dir to filename.tar.gz file
    tar_file = shutil.make_archive(file_path, format=format, root_dir=file_path)

    # Remove filename dir
    if os.path.exists(file_path):
        shutil.rmtree(file_path)

    # filename.tar.gz rename to filename
    os.rename(tar_file, file_path)


@contextlib.contextmanager
def unpack_archive_with_buffer(buffer, format="gztar"):
    """Unpack archive with archive buffer
    After the call is finished, the archive file and directory will be deleted.

    Implementation process:
        1. create 'tempfile' in '~/tmp/' and directory
        2. 'buffer' write to 'tempfile'
        3. unpack archive file('tempfile')
        4. user does something with file_path('tempfile/')
        5. remove 'tempfile' and 'tempfile directory'

    :param buffer: bytes
    :param format: archive format: one of "zip", "tar", "gztar", "bztar", or "xztar"
    :return: unpack archive directory path

    Usage::

        >>> # The following code is to print all the file names in 'test_unpack.tar.gz'
        >>> with open('test_unpack.tar.gz') as fp:
        ...     buffer = fp.read()
        ...
        >>> with unpack_archive_with_buffer(buffer) as temp_dir:
        ...     for f_n in os.listdir(temp_dir):
        ...         print(f_n)
        ...

    """
    temp_dir = os.path.expanduser("~/tmp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=temp_dir) as fp:
        fp.write(buffer)
        file_path = fp.name

    try:
        tar_file = file_path + ".tar.gz"
        os.rename(file_path, tar_file)
        # Create dir
        os.makedirs(file_path)
        shutil.unpack_archive(tar_file, format=format, extract_dir=file_path)

        # Return temp dir
        yield file_path

    except Exception as e:
        log.error(str(e))
    finally:
        # Remove temp tar file
        if os.path.exists(tar_file):
            os.unlink(tar_file)

        # Remove temp model dir
        if os.path.exists(file_path):
            shutil.rmtree(file_path)


@contextlib.contextmanager
def get_tmp_file_with_buffer(buffer):
    temp_dir = os.path.expanduser("~/tmp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with tempfile.NamedTemporaryFile("wb", delete=True, dir=temp_dir) as fp:
        fp.write(buffer)
        file_path = fp.name
        yield file_path


def remove_repeat_field(fields):
    """remove repeat field

    :param fields: list; features fields
    :return: list
    """
    fields = copy.deepcopy(fields)
    _fields = set(fields)
    return sorted(_fields, key=fields.index)


def remove_fields_space(fields: [list, str, tuple]):
    """remove fields space

    :param fields: features fields
    :return: list or str
    """
    if isinstance(fields, str):
        return fields.replace(" ", "")
    return [i.replace(" ", "") for i in fields if isinstance(i, str)]


def normalize_cache_fields(fields: [list, tuple]):
    """normalize cache fields

    :param fields: features fields
    :return: list
    """
    return sorted(remove_repeat_field(remove_fields_space(fields)))


def normalize_cache_instruments(instruments):
    """normalize cache instruments

    :return: list or dict
    """
    if isinstance(instruments, (list, tuple, pd.Index, np.ndarray)):
        instruments = sorted(list(instruments))
    else:
        # dict type stockpool
        if "market" in instruments:
            pass
        else:
            instruments = {k: sorted(v) for k, v in instruments.items()}
    return instruments


def is_tradable_date(cur_date):
    """judgy whether date is a tradable date
    ----------
    date : pandas.Timestamp
        current date
    """
    from ..data import D

    return str(cur_date.date()) == str(D.calendar(start_time=cur_date, future=True)[0].date())


def get_date_range(trading_date, left_shift=0, right_shift=0, future=False):
    """get trading date range by shift

    Parameters
    ----------
    trading_date: pd.Timestamp
    left_shift: int
    right_shift: int
    future: bool

    """

    from ..data import D

    start = get_date_by_shift(trading_date, left_shift, future=future)
    end = get_date_by_shift(trading_date, right_shift, future=future)

    calendar = D.calendar(start, end, future=future)
    return calendar


def get_date_by_shift(trading_date, shift, future=False, clip_shift=True):
    """get trading date with shift bias wil cur_date
        e.g. : shift == 1,  return next trading date
               shift == -1, return previous trading date
    ----------
    trading_date : pandas.Timestamp
        current date
    shift : int
    clip_shift: bool

    """
    from qlib.data import D

    cal = D.calendar(future=future)
    if pd.to_datetime(trading_date) not in list(cal):
        raise ValueError("{} is not trading day!".format(str(trading_date)))
    _index = bisect.bisect_left(cal, trading_date)
    shift_index = _index + shift
    if shift_index < 0 or shift_index >= len(cal):
        if clip_shift:
            shift_index = np.clip(shift_index, 0, len(cal) - 1)
        else:
            raise IndexError(f"The shift_index({shift_index}) of the trading day ({trading_date}) is out of range")
    return cal[shift_index]


def get_next_trading_date(trading_date, future=False):
    """get next trading date
    ----------
    cur_date : pandas.Timestamp
        current date
    """
    return get_date_by_shift(trading_date, 1, future=future)


def get_pre_trading_date(trading_date, future=False):
    """get previous trading date
    ----------
    date : pandas.Timestamp
        current date
    """
    return get_date_by_shift(trading_date, -1, future=future)


def transform_end_date(end_date=None, freq="day"):
    """get previous trading date
    If end_date is -1, None, or end_date is greater than the maximum trading day, the last trading date is returned.
    Otherwise, returns the end_date
    ----------
    end_date: str
        end trading date
    date : pandas.Timestamp
        current date
    """
    from ..data import D

    last_date = D.calendar(freq=freq)[-1]
    if end_date is None or (str(end_date) == "-1") or (pd.Timestamp(last_date) < pd.Timestamp(end_date)):
        log.warning(
            "\nInfo: the end_date in the configuration file is {}, "
            "so the default last date {} is used.".format(end_date, last_date)
        )
        end_date = last_date
    return end_date


def get_date_in_file_name(file_name):
    """Get the date(YYYY-MM-DD) written in file name
    Parameter
            file_name : str
       :return
            date : str
                'YYYY-MM-DD'
    """
    pattern = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
    date = re.search(pattern, str(file_name)).group()
    return date


def split_pred(pred, number=None, split_date=None):
    """split the score file into two part
    Parameter
    ---------
        pred : pd.DataFrame (index:<instrument, datetime>)
            A score file of stocks
        number: the number of dates for pred_left
        split_date: the last date of the pred_left
    Return
    -------
        pred_left : pd.DataFrame (index:<instrument, datetime>)
            The first part of original score file
        pred_right : pd.DataFrame (index:<instrument, datetime>)
            The second part of original score file
    """
    if number is None and split_date is None:
        raise ValueError("`number` and `split date` cannot both be None")
    dates = sorted(pred.index.get_level_values("datetime").unique())
    dates = list(map(pd.Timestamp, dates))
    if split_date is None:
        date_left_end = dates[number - 1]
        date_right_begin = dates[number]
        date_left_start = None
    else:
        split_date = pd.Timestamp(split_date)
        date_left_end = split_date
        date_right_begin = split_date + pd.Timedelta(days=1)
        if number is None:
            date_left_start = None
        else:
            end_idx = bisect.bisect_right(dates, split_date)
            date_left_start = dates[end_idx - number]
    pred_temp = pred.sort_index()
    pred_left = pred_temp.loc(axis=0)[:, date_left_start:date_left_end]
    pred_right = pred_temp.loc(axis=0)[:, date_right_begin:]
    return pred_left, pred_right


def can_use_cache():
    res = True
    r = get_redis_connection()
    try:
        r.client()
    except redis.exceptions.ConnectionError:
        res = False
    finally:
        r.close()
    return res


def exists_qlib_data(qlib_dir):
    qlib_dir = Path(qlib_dir).expanduser()
    if not qlib_dir.exists():
        return False

    calendars_dir = qlib_dir.joinpath("calendars")
    instruments_dir = qlib_dir.joinpath("instruments")
    features_dir = qlib_dir.joinpath("features")
    # check dir
    for _dir in [calendars_dir, instruments_dir, features_dir]:
        if not (_dir.exists() and list(_dir.iterdir())):
            return False
    # check calendar bin
    for _calendar in calendars_dir.iterdir():
        if not list(features_dir.rglob(f"*.{_calendar.name.split('.')[0]}.bin")):
            return False

    # check instruments
    code_names = set(map(lambda x: x.name.lower(), features_dir.iterdir()))
    _instrument = instruments_dir.joinpath("all.txt")
    miss_code = set(pd.read_csv(_instrument, sep="\t", header=None).loc[:, 0].apply(str.lower)) - set(code_names)
    if miss_code and any(map(lambda x: "sht" not in x, miss_code)):
        return False

    return True


def check_qlib_data(qlib_config):
    inst_dir = Path(qlib_config["provider_uri"]).joinpath("instruments")
    for _p in inst_dir.glob("*.txt"):
        try:
            assert len(pd.read_csv(_p, sep="\t", nrows=0, header=None).columns) == 3, (
                f"\nThe {str(_p.resolve())} of qlib data is not equal to 3 columns:"
                f"\n\tIf you are using the data provided by qlib: "
                f"https://qlib.readthedocs.io/en/latest/component/data.html#qlib-format-dataset"
                f"\n\tIf you are using your own data, please dump the data again: "
                f"https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format"
            )
        except AssertionError:
            raise


def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    """
    make the df index sorted

    df.sort_index() will take a lot of time even when `df.is_lexsorted() == True`
    This function could avoid such case

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame:
        sorted dataframe
    """
    idx = df.index if axis == 0 else df.columns
    if idx.is_monotonic_increasing:
        return df
    else:
        return df.sort_index(axis=axis)


def flatten_dict(d, parent_key="", sep="."):
    """flatten_dict.
        >>> flatten_dict({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]})
        >>> {'a': 1, 'c.a': 2, 'c.b.x': 5, 'd': [1, 2, 3], 'c.b.y': 10}

    Parameters
    ----------
    d :
        d
    parent_key :
        parent_key
    sep :
        sep
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


#################### Wrapper #####################
class Wrapper:
    """Wrapper class for anything that needs to set up during qlib.init"""

    def __init__(self):
        self._provider = None

    def register(self, provider):
        self._provider = provider

    def __repr__(self):
        return "{name}(provider={provider})".format(name=self.__class__.__name__, provider=self._provider)

    def __getattr__(self, key):
        if self._provider is None:
            raise AttributeError("Please run qlib.init() first using qlib")
        return getattr(self._provider, key)


def register_wrapper(wrapper, cls_or_obj, module_path=None):
    """register_wrapper

    :param wrapper: A wrapper.
    :param cls_or_obj:  A class or class name or object instance.
    """
    if isinstance(cls_or_obj, str):
        module = get_module_by_module_path(module_path)
        cls_or_obj = getattr(module, cls_or_obj)
    obj = cls_or_obj() if isinstance(cls_or_obj, type) else cls_or_obj
    wrapper.register(obj)


def load_dataset(path_or_obj):
    """load dataset from multiple file formats"""
    if isinstance(path_or_obj, pd.DataFrame):
        return path_or_obj
    if not os.path.exists(path_or_obj):
        raise ValueError(f"file {path_or_obj} doesn't exist")
    _, extension = os.path.splitext(path_or_obj)
    if extension == ".h5":
        return pd.read_hdf(path_or_obj)
    elif extension == ".pkl":
        return pd.read_pickle(path_or_obj)
    elif extension == ".csv":
        return pd.read_csv(path_or_obj, parse_dates=True, index_col=[0, 1])
    raise ValueError(f"unsupported file type `{extension}`")


def code_to_fname(code: str):
    """stock code to file name

    Parameters
    ----------
    code: str
    """
    # NOTE: In windows, the following name is I/O device, and the file with the corresponding name cannot be created
    # reference: https://superuser.com/questions/86999/why-cant-i-name-a-folder-or-file-con-in-windows
    replace_names = ["CON", "PRN", "AUX", "NUL"]
    replace_names += [f"COM{i}" for i in range(10)]
    replace_names += [f"LPT{i}" for i in range(10)]

    prefix = "_qlib_"
    if str(code).upper() in replace_names:
        code = prefix + str(code)

    return code


def fname_to_code(fname: str):
    """file name to stock code

    Parameters
    ----------
    fname: str
    """
    prefix = "_qlib_"
    if fname.startswith(prefix):
        fname = fname.lstrip(prefix)
    return fname

########################## Sample ############################
def sample_calendar_bac(calendar_raw, freq_raw, freq_sam):
    """
    freq_raw : "min" or "day"
    """
    freq_raw = "1" + freq_raw if re.match("^[0-9]", freq_raw) is None else freq_raw
    freq_sam = "1" + freq_sam if re.match("^[0-9]", freq_sam) is None else freq_sam

    if freq_sam.endswith(("minute", "min")):
        def cal_next_sam_minute(x, sam_minutes):
            hour = x.hour
            minute = x.minute
            if 9 <= hour <= 11:
                minute_index = (11 - hour)*60 + 30 - minute + 120
            elif 13 <= hour <= 15:
                minute_index = (15 - hour)*60 - minute
            else:
                raise ValueError("calendar hour must be in [9, 11] or [13, 15]")
            
            minute_index = minute_index // sam_minutes * sam_minutes

            if 0 <= minute_index < 120:
                return 15 - (minute_index + 59) // 60, (120 - minute_index) % 60
            elif 120 <= minute_index < 240:
                return 11 - (minute_index - 120 + 29) // 60, (240 - minute_index + 30) % 60
            else:
                raise ValueError("calendar minute_index error")

        sam_minutes = int(freq_sam[:-3]) if freq_sam.endswith("min") else int(freq_sam[:-6])

        if not freq_raw.endswith(("minute", "min")):
            raise ValueError("when sampling minute calendar, freq of raw calendar must be minute or min")
        else:
            raw_minutes = int(freq_raw[:-3]) if freq_raw.endswith("min") else int(freq_raw[:-6])
            if raw_minutes > sam_minutes:
                raise ValueError("raw freq must be higher than sample freq")

        _calendar_minute = np.unique(list(map(lambda x: pd.Timestamp(x.year, x.month, x.day, *cal_next_sam_minute(x, sam_minutes), 59), calendar_raw)))
        return _calendar_minute
    else:

        _calendar_day = np.unique(list(map(lambda x: pd.Timestamp(x.year, x.month, x.day, 23, 59, 59), calendar_raw)))
        if freq_sam.endswith(("day", "d")):
            sam_days = int(freq_sam[:-1]) if freq_sam.endswith("d") else int(freq_sam[:-3])
            return _calendar_day[(len(_calendar_day) + sam_days - 1)%sam_days::sam_days]

        elif freq_sam.endswith(("week", "w")):
            sam_weeks = int(freq_sam[:-1]) if freq_sam.endswith("w") else int(freq_sam[:-4])
            _day_in_week = np.array(list(map(lambda x: x.dayofweek, _calendar_day)))
            _calendar_week = _calendar_day[np.ediff1d(_day_in_week[::-1], to_begin=1)[::-1] > 0]
            return _calendar_week[(len(_calendar_week) + sam_weeks - 1)%sam_weeks::sam_weeks]

        elif freq_sam.endswith(("month", "m")):
            sam_months = int(freq_sam[:-1]) if freq_sam.endswith("m") else int(freq_sam[:-5])
            _day_in_month = np.array(list(map(lambda x: x.day, _calendar_day)))
            _calendar_month = _calendar_day[np.ediff1d(_day_in_month[::-1], to_begin=1)[::-1] > 0]
            return _calendar_month[(len(_calendar_month) + sam_months - 1)%sam_months::sam_months]
        else:
            raise ValueError("sample freq must be xmin, xd, xw, xm")

def sample_calendar(calendar_raw, freq_raw, freq_sam):
    """
    freq_raw : "min" or "day"
    """
    freq_raw = "1" + freq_raw if re.match("^[0-9]", freq_raw) is None else freq_raw
    freq_sam = "1" + freq_sam if re.match("^[0-9]", freq_sam) is None else freq_sam
    if not len(calendar_raw):
        return calendar_raw
    if freq_sam.endswith(("minute", "min")):
        def cal_next_sam_minute(x, sam_minutes):
            hour = x.hour
            minute = x.minute
            if (hour == 9 and minute >= 30) or (9 < hour < 11) or (hour == 11 and minute < 30):
                minute_index = (hour - 9)*60 + minute - 30
            elif 13 <= hour < 15:
                minute_index = (hour - 13)*60 + minute + 120
            else:
                raise ValueError("calendar hour must be in [9, 11] or [13, 15]")
            
            minute_index = minute_index // sam_minutes * sam_minutes

            if 0 <= minute_index < 120:
                return 9 + (minute_index + 30) // 60, (minute_index + 30) % 60
            elif 120 <= minute_index < 240:
                return 13 + (minute_index - 120) // 60, (minute_index - 120) % 60
            else:
                raise ValueError("calendar minute_index error")
        sam_minutes = int(freq_sam[:-3]) if freq_sam.endswith("min") else int(freq_sam[:-6])
        if not freq_raw.endswith(("minute", "min")):
            raise ValueError("when sampling minute calendar, freq of raw calendar must be minute or min")
        else:
            raw_minutes = int(freq_raw[:-3]) if freq_raw.endswith("min") else int(freq_raw[:-6])
            if raw_minutes > sam_minutes:
                raise ValueError("raw freq must be higher than sample freq")
        _calendar_minute = np.unique(list(map(lambda x: pd.Timestamp(x.year, x.month, x.day, *cal_next_sam_minute(x, sam_minutes), 0), calendar_raw)))
        if calendar_raw[0] > _calendar_minute[0]:
            _calendar_minute[0] = calendar_raw[0]
        return _calendar_minute
    else:
        _calendar_day = np.unique(list(map(lambda x: pd.Timestamp(x.year, x.month, x.day, 0, 0, 0), calendar_raw)))
        if freq_sam.endswith(("day", "d")):
            sam_days = int(freq_sam[:-1]) if freq_sam.endswith("d") else int(freq_sam[:-3])
            return _calendar_day[::sam_days]

        elif freq_sam.endswith(("week", "w")):
            sam_weeks = int(freq_sam[:-1]) if freq_sam.endswith("w") else int(freq_sam[:-4])
            _day_in_week = np.array(list(map(lambda x: x.dayofweek, _calendar_day)))
            _calendar_week = _calendar_day[np.ediff1d(_day_in_week, to_begin=-1) < 0]
            return _calendar_week[::sam_weeks]

        elif freq_sam.endswith(("month", "m")):
            sam_months = int(freq_sam[:-1]) if freq_sam.endswith("m") else int(freq_sam[:-5])
            _day_in_month = np.array(list(map(lambda x: x.day, _calendar_day)))
            _calendar_month = _calendar_day[np.ediff1d(_day_in_month, to_begin=-1) < 0]
            return _calendar_month[::sam_months]
        else:
            raise ValueError("sample freq must be xmin, xd, xw, xm")
            
def get_sample_freq_calendar(start_time, end_time, freq):
    try:
        _calendar = D.calendar(start_time=start_time, end_time=end_time, freq=freq)
    except ValueError:
        if freq.endswith(("m", "month", "w", "week", "d", "day")):
            try:
                _calendar = D.calendar(start_time=self.start_time, end_time=self.end_time, freq="min", freq_sam=freq)
            except ValueError:
                _calendar = D.calendar(start_time=self.start_time, end_time=self.end_time, freq="day", freq_sam=freq)
        elif freq.endswith(("min", "minute")):
            _calendar = D.calendar(start_time=self.start_time, end_time=self.end_time, freq="min", freq_sam=freq)
        else:
            raise ValueError(f"freq {freq} is not supported")
    return _calendar

def sample_feature(feature, instruments=None, start_time=None, end_time=None, fields=None, method=None, method_kwargs={}):
    if instruments and type(instruments) is not list:
        instruments = [instruments]
    if fields and type(fields) is not list:
        fields = [fields]
    selector_inst = slice(None) if instruments is None else instruments
    selector_datetime = slice(start_time, end_time)
    if fields is not None and type(fields) is not list:
        fields = [fields]
    selector_fields = slice(None) if fields is None else fields
    feature = feature.loc[(selector_inst, selector_datetime), selector_fields]
    if method:
        return getattr(feature.groupby(level="instrument"), method)(**method_kwargs)
    else:
        return feature

