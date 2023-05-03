# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
TODO:
- A more well-designed PIT database is required.
    - seperated insert, delete, update, query operations are required.
"""

import shutil
import struct
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, List, Union

import fire
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from qlib.config import C
from qlib.utils import fname_to_code, get_period_offset


class DumpPitData:
    PIT_DIR_NAME = "financial"
    PIT_CSV_SEP = ","
    DATA_FILE_SUFFIX = ".data"
    INDEX_FILE_SUFFIX = ".index"

    INTERVAL_quarterly = "quarterly"
    INTERVAL_annual = "annual"
    INTERVAL_INDEFINITE = "indefinite"
    INTERVAL_MONTHLY = "monthly"

    PERIOD_DTYPE = C.pit_record_type["period"]
    INDEX_DTYPE = C.pit_record_type["index"]
    DATA_DTYPE = "".join(
        [
            C.pit_record_type["date"],
            C.pit_record_type["period"],
            C.pit_record_type["value"],
            C.pit_record_type["index"],
        ]
    )

    NA_INDEX = C.pit_record_nan["index"]

    INDEX_DTYPE_SIZE = struct.calcsize(INDEX_DTYPE)
    PERIOD_DTYPE_SIZE = struct.calcsize(PERIOD_DTYPE)
    DATA_DTYPE_SIZE = struct.calcsize(DATA_DTYPE)

    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        csv_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "quarterly",
        max_workers: int = 16,
        date_column_name: str = "date",
        period_column_name: str = "period",
        value_column_name: str = "value",
        field_column_name: str = "field",
        file_suffix: str = ".csv",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        csv_path: str
            stock data path or directory
        qlib_dir: str
            qlib(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup qlib_dir to backup_dir
        freq: str, default "quarterly"
            data frequency
        max_workers: int, default None
            number of threads
        date_column_name: str, default "date"
            the name of the date field in the csv
        file_suffix: str, default ".csv"
            file suffix
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        csv_path = Path(csv_path).expanduser()
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.file_suffix = file_suffix
        self.csv_files = sorted(csv_path.glob(f"*{self.file_suffix}") if csv_path.is_dir() else [csv_path])
        if limit_nums is not None:
            self.csv_files = self.csv_files[: int(limit_nums)]
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())

        self.works = max_workers
        self.date_column_name = date_column_name
        self.period_column_name = period_column_name
        self.value_column_name = value_column_name
        self.field_column_name = field_column_name

        self._mode = self.ALL_MODE

    def _backup_qlib_dir(self, target_dir: Path):
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def get_source_data(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(str(file_path.resolve()), low_memory=False)
        df[self.value_column_name] = df[self.value_column_name].astype("float32")
        df[self.date_column_name] = df[self.date_column_name].str.replace("-", "").astype("int32")
        # df.drop_duplicates([self.date_field_name], inplace=True)
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        return fname_to_code(file_path.name[: -len(self.file_suffix)].strip().lower())

    def get_dump_fields(self, df: Iterable[str]) -> Iterable[str]:
        return (
            set(self._include_fields)
            if self._include_fields
            else set(df[self.field_column_name]) - set(self._exclude_fields)
            if self._exclude_fields
            else set(df[self.field_column_name])
        )

    def get_filenames(self, symbol, field, interval):
        dir_name = self.qlib_dir.joinpath(self.PIT_DIR_NAME, symbol)
        dir_name.mkdir(parents=True, exist_ok=True)
        return (
            dir_name.joinpath(f"{field}_{interval[0]}{self.DATA_FILE_SUFFIX}".lower()),
            dir_name.joinpath(f"{field}_{interval[0]}{self.INDEX_FILE_SUFFIX}".lower()),
        )

    def _dump_pit(
        self,
        file_path: str,
        interval: str = "quarterly",
        overwrite: bool = False,
    ):
        """
        dump data as the following format:
            `/path/to/<field>.data`
                [date, period, value, _next]
                [date, period, value, _next]
                [...]
            `/path/to/<field>.index`
                [first_year, index, index, ...]

        `<field.data>` contains the data as the point-in-time (PIT) order: `value` of `period`
        is published at `date`, and its successive revised value can be found at `_next` (linked list).

        `<field>.index` contains the index of value for each period (quarter or year). To save
        disk space, we only store the `first_year` as its followings periods can be easily infered.

        Parameters
        ----------
        file_path: str
            path of dump file
        interval: str
            data interval
        overwrite: bool
            whether overwrite existing data or update only
        """
        symbol = self.get_symbol_from_file(file_path)
        df = self.get_source_data(file_path)
        if df.empty:
            logger.warning(f"{symbol} file is empty")
            return
        for field in self.get_dump_fields(df):
            df_sub = df.query(f'{self.field_column_name}=="{field}"').sort_values(self.date_column_name)
            if df_sub.empty:
                logger.warning(f"field {field} of {symbol} is empty")
                continue
            data_file, index_file = self.get_filenames(symbol, field, interval)

            if data_file.exists() ^ index_file.exists():
                logger.warning(f"{data_file} and {index_file} should be both exist or not. Forcing to overwrite.")
            overwrite = overwrite or not (data_file.exists() and index_file.exists())

            ## calculate first & last period
            start_year = df_sub[self.period_column_name].min()
            end_year = df_sub[self.period_column_name].max()
            if interval[0] in {"q", "m"}:
                start_year //= 100
                end_year //= 100

            # adjust `first_year` if existing data found
            if not overwrite:
                with open(index_file, "rb") as fi:
                    (first_year,) = struct.unpack(self.PERIOD_DTYPE, fi.read(self.PERIOD_DTYPE_SIZE))
                    n_years = len(fi.read()) // self.INDEX_DTYPE_SIZE
                    if interval[0] == "q":
                        n_years //= 4
                    elif interval[0] == "m":
                        n_years //= 12
                    start_year = first_year + n_years
            else:
                with open(index_file, "wb") as f:
                    f.write(struct.pack(self.PERIOD_DTYPE, start_year))
                first_year = start_year

            # if data already exists, continue to the next field
            # if start_year > end_year:
            #     logger.warning(f"{symbol}-{field} data already exists, continue to the next field")
            #     continue

            # dump index filled with NA
            with open(index_file, "ab") as fi:
                for year in range(start_year, end_year + 1):
                    if interval[0] == "q":
                        fi.write(struct.pack(self.INDEX_DTYPE * 4, *[self.NA_INDEX] * 4))
                    elif interval[0] == "m":
                        fi.write(struct.pack(self.INDEX_DTYPE * 12, *[self.NA_INDEX] * 12))
                    else:
                        fi.write(struct.pack(self.INDEX_DTYPE, self.NA_INDEX))

            # if data already exists, remove overlapped data
            truncated_size = self.NA_INDEX
            if not overwrite:
                start_date = df_sub[self.date_column_name].min()
                truncated_periods = set()
                with open(data_file, "rb+") as fd, open(index_file, "rb+") as fi:
                    try:
                        fd.seek(-self.DATA_DTYPE_SIZE, 2)
                    except OSError:
                        raise
                    seek_back_bytes = -2 * self.DATA_DTYPE_SIZE
                    while fd.tell() >= 0:
                        last_date, period, *_ = struct.unpack(self.DATA_DTYPE, fd.read(self.DATA_DTYPE_SIZE))
                        if last_date < start_date:
                            truncated_size = fd.tell()
                            fd.truncate()
                            break
                        truncated_periods.add(period)
                        if fd.tell() + seek_back_bytes < 0:
                            truncated_size = 0
                            fd.truncate(0)
                            break
                        fd.seek(seek_back_bytes, 1)
                with open(index_file, "rb+") as fi:
                    for period in truncated_periods:
                        offset = get_period_offset(first_year, period, interval[0])
                        fi.seek(self.PERIOD_DTYPE_SIZE + self.INDEX_DTYPE_SIZE * offset)
                        (cur_index,) = struct.unpack(self.INDEX_DTYPE, fi.read(self.INDEX_DTYPE_SIZE))
                        if cur_index!= self.NA_INDEX and cur_index >= truncated_size:
                            fi.seek(-self.INDEX_DTYPE_SIZE, 1)
                            fi.write(struct.pack(self.INDEX_DTYPE, self.NA_INDEX))

            # otherwise,
            # 1) truncate existing file or create a new file with `wb+` if `overwrite` is True,
            # 2) or append existing file or create a new file with `ab+` if not overwrite
            else:
                with open(data_file, "wb+" if overwrite else "ab+"):
                    pass

            with open(data_file, "rb+") as fd, open(index_file, "rb+") as fi:
                # update index if needed
                for i, row in df_sub.iterrows():
                    # get index
                    offset = get_period_offset(first_year, row.period, interval[0])

                    fi.seek(self.PERIOD_DTYPE_SIZE + self.INDEX_DTYPE_SIZE * offset)
                    (cur_index,) = struct.unpack(self.INDEX_DTYPE, fi.read(self.INDEX_DTYPE_SIZE))


                    # Case I: new data => update `_next` with current index
                    if cur_index == self.NA_INDEX:
                        fi.seek(self.PERIOD_DTYPE_SIZE + self.INDEX_DTYPE_SIZE * offset)
                        fi.write(struct.pack(self.INDEX_DTYPE, fd.tell()))
                    # Case II: previous data exists => find and update the last `_next`
                    else:
                        _cur_fd = fd.seek(0, 2)
                        prev_index = self.NA_INDEX
                        while cur_index < truncated_size:  # NOTE: first iter always != NA_INDEX
                            fd.seek(cur_index + self.DATA_DTYPE_SIZE - self.INDEX_DTYPE_SIZE)
                            prev_index = cur_index
                            (cur_index,) = struct.unpack(self.INDEX_DTYPE, fd.read(self.INDEX_DTYPE_SIZE))
                        fd.seek(prev_index + self.DATA_DTYPE_SIZE - self.INDEX_DTYPE_SIZE)
                        fd.write(struct.pack(self.INDEX_DTYPE, _cur_fd))  # NOTE: add _next pointer
                        fd.seek(_cur_fd)

                    # dump data
                    fd.write(struct.pack(self.DATA_DTYPE, row.date, row.period, row.value, self.NA_INDEX))

    def dump(self, interval="quarterly", overwrite=False):
        logger.info("start dump pit data......")
        _dump_func = partial(self._dump_pit, interval=interval, overwrite=overwrite)

        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.csv_files):
                    p_bar.update()

    def __call__(self, *args, **kwargs):
        self.dump(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(DumpPitData)
