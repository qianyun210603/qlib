#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import sys
import fire
import glob

import pandas as pd
import ruamel.yaml as yaml
import shutil
import signal
import inspect
import functools
import statistics
from datetime import datetime
from pathlib import Path
from operator import xor
from pprint import pprint
from loguru import logger
import qlib
from qlib.config import C
from qlib.workflow import R

from qlib.workflow.cli import sys_config
from qlib.model.trainer import _log_task_info, _exe_task


# decorator to check the arguments
def only_allow_defined_args(function_to_decorate):
    @functools.wraps(function_to_decorate)
    def _return_wrapped(*args, **kwargs):
        """Internal wrapper function."""
        argspec = inspect.getfullargspec(function_to_decorate)
        valid_names = set(argspec.args + argspec.kwonlyargs)
        if "self" in valid_names:
            valid_names.remove("self")
        for arg_name in kwargs:
            if arg_name not in valid_names:
                raise ValueError("Unknown argument seen '%s', expected: [%s]" % (arg_name, ", ".join(valid_names)))
        return function_to_decorate(*args, **kwargs)

    return _return_wrapped


# function to handle ctrl z and ctrl c
def handler(_, __):
    os.system("kill -9 %d" % os.getpid())


signal.signal(signal.SIGINT, handler)


# function to calculate the mean and std of a list in the results dictionary
def cal_mean_std(results) -> dict:
    mean_std = dict()
    for fn in results:
        try:
            mean_std[fn] = dict()
            for metric in results[fn]:
                mean = statistics.mean(results[fn][metric]) if len(results[fn][metric]) > 1 else results[fn][metric][0]
                std = statistics.stdev(results[fn][metric]) if len(results[fn][metric]) > 1 else 0
                mean_std[fn][metric] = [mean, std]
        except Exception:
            print(fn)
    return mean_std


def workflow(config_path, experiment_name="workflow", uri_folder="mlruns", force_rerun=False, target_run_num=1):
    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    # config the `sys` section
    sys_config(config, config_path)

    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(config_path).parent.parent.joinpath(uri_folder))
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    if "experiment_name" in config:
        experiment_name = config["experiment_name"]

    this_exp = R.get_exp(experiment_name=experiment_name)
    existing_recorders = this_exp.list_recorders()
    num_valid_results = 0
    for rec_id, rec in existing_recorders.items():
        if force_rerun:
            this_exp.delete_recorder(recorder_id=rec_id)
        elif rec.status != 'FINISHED':
            logger.info(f"delete stale record {rec_id}")
            this_exp.delete_recorder(recorder_id=rec_id)
        elif num_valid_results >= target_run_num:
            logger.info(f"already have enough records, delete {rec_id}")
            this_exp.delete_recorder(recorder_id=rec_id)
        else:
            num_valid_results += 1

    while num_valid_results < target_run_num:
        with R.start(experiment_name=experiment_name):
            recorder = R.get_recorder()
            logger.info(
                f"Already have {num_valid_results} records, {target_run_num-num_valid_results} to run."
            )
            logger.info(f"Record id for this run: {recorder.id}")
            _log_task_info(config['task'])
            _exe_task(config['task'])
            recorder.save_objects(config=config)
            num_valid_results += 1


# function to get all the folders benchmark folder
def get_all_folders(models, exclude) -> dict:
    folders = dict()
    if isinstance(models, str):
        model_list = models.split(",")
        models = [m.lower().strip("[ ]") for m in model_list]
    elif isinstance(models, list):
        models = [m.lower() for m in models]
    elif models is None:
        models = [f.name.lower() for f in os.scandir("benchmarks")]
    else:
        raise ValueError("Input models type is not supported. Please provide str or list without space.")
    for f in os.scandir("benchmarks"):
        add = xor(bool(f.name.lower() in models), bool(exclude))
        if add:
            path = Path("benchmarks") / f.name
            folders[f.name] = str(path.resolve())
    return folders


# function to get all the files under the model folder
def get_all_files(folder_path, dataset, universe="") -> (str, str):
    if universe != "":
        universe = f"_{universe}"
    yaml_file = list(Path(f"{folder_path}").glob(f"*{dataset}{universe}.yaml"))
    if len(yaml_file) == 0:
        return None
    else:
        return yaml_file[0]


# function to retrieve all the results
def get_all_results(folders) -> dict:
    results = dict()
    for fn in folders:
        try:
            exp = R.get_exp(experiment_name=fn, create=False)
        except ValueError:
            # No experiment results
            continue
        recorders = exp.list_recorders()
        result = dict()
        result["annualized_return_with_cost"] = list()
        result["information_ratio_with_cost"] = list()
        result["max_drawdown_with_cost"] = list()
        result["ic"] = list()
        result["icir"] = list()
        result["rank_ic"] = list()
        result["rank_icir"] = list()
        for recorder_id in recorders:
            if recorders[recorder_id].status == "FINISHED":
                recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=fn)
                metrics = recorder.list_metrics()
                if "1day.excess_return_with_cost.annualized_return" not in metrics:
                    print(f"{recorder_id} is skipped due to incomplete result")
                    continue
                result["annualized_return_with_cost"].append(metrics["1day.excess_return_with_cost.annualized_return"])
                result["information_ratio_with_cost"].append(metrics["1day.excess_return_with_cost.information_ratio"])
                result["max_drawdown_with_cost"].append(metrics["1day.excess_return_with_cost.max_drawdown"])
                result["ic"].append(metrics["IC"])
                result["icir"].append(metrics["ICIR"])
                result["rank_ic"].append(metrics["Rank IC"])
                result["rank_icir"].append(metrics["Rank ICIR"])
        results[fn] = result
    return results


# function to generate and save markdown table
def gen_and_save_md_table(metrics, dataset, base_path='.'):
    table = "| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for fn in metrics:
        ic = metrics[fn]["ic"]
        icir = metrics[fn]["icir"]
        ric = metrics[fn]["rank_ic"]
        ricir = metrics[fn]["rank_icir"]
        ar = metrics[fn]["annualized_return_with_cost"]
        ir = metrics[fn]["information_ratio_with_cost"]
        md = metrics[fn]["max_drawdown_with_cost"]
        table += f"| {fn} | {dataset} | {ic[0]:5.4f}±{ic[1]:2.2f} | {icir[0]:5.4f}±{icir[1]:2.2f}| {ric[0]:5.4f}±{ric[1]:2.2f} | {ricir[0]:5.4f}±{ricir[1]:2.2f} | {ar[0]:5.4f}±{ar[1]:2.2f} | {ir[0]:5.4f}±{ir[1]:2.2f}| {md[0]:5.4f}±{md[1]:2.2f} |\n"
    pprint(table)
    with open(Path(base_path).resolve().joinpath("table.md"), "w") as f:
        f.write(table)
    return table


# read yaml, remove seed kwargs of model, and then save file in the temp_dir
def gen_yaml_files_from_example_templates(
    yaml_path, temp_dir, provider_uri, train, valid, test
):
    yaml_path = Path(yaml_path).expanduser()
    with open(yaml_path, "r") as fp:
        config = yaml.safe_load(fp)
    file_name = yaml_path.name if config['market'] in yaml_path.name else \
        yaml_path.stem.rstrip("_full") + '_' + config['market'] + yaml_path.suffix

    temp_path = Path(temp_dir).expanduser().joinpath(file_name)
    if temp_path.exists():
        return temp_path

    try:
        del config["task"]["model"]["kwargs"]["seed"]
    except KeyError:
        pass

    if 'sys' in config and 'rel_path' in config['sys']:
        config['sys'].setdefault('path', []).extend(
            str((yaml_path.parent / p).resolve().absolute()) for p in config['sys']['rel_path']
        )
        del config['sys']['rel_path']


    config['qlib_init']['provider_uri'] = provider_uri

    config['task']['dataset']['kwargs']['handler']['kwargs']['start_time'] = train[0]
    config['task']['dataset']['kwargs']['handler']['kwargs']['end_time'] = test[1]
    if 'valid' in config['task']['dataset']['kwargs']['segments']:
        config['task']['dataset']['kwargs']['handler']['kwargs']['fit_start_time'] = train[0]
        config['task']['dataset']['kwargs']['handler']['kwargs']['fit_end_time'] = train[1]
        config['task']['dataset']['kwargs']['segments']['train'] = train
        config['task']['dataset']['kwargs']['segments']['valid'] = valid
        config['task']['dataset']['kwargs']['segments']['test'] = test
    else:
        config['task']['dataset']['kwargs']['handler']['kwargs']['fit_start_time'] = train[0]
        config['task']['dataset']['kwargs']['handler']['kwargs']['fit_end_time'] = valid[1]
        config['task']['dataset']['kwargs']['segments']['train'] = [train[0], valid[1]]
        config['task']['dataset']['kwargs']['segments']['test'] = test

    config['task']['record'][-1]['kwargs']['config']['backtest']['start_time'] = \
        (pd.Timestamp(test[0]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    config['task']['record'][-1]['kwargs']['config']['backtest']['end_time'] = test[1]
    if 'infer_processors' in config['task']['dataset']['kwargs']['handler']['kwargs']:
        config['task']['dataset']['kwargs']['handler']['kwargs']['infer_processors'] = \
            [
                p for p in config['task']['dataset']['kwargs']['handler']['kwargs']['infer_processors']
                if p['class'] != 'FilterCol'
            ]

    if config['task']['model']['module_path'].endswith('_ts'):
        config['task']['model']['module_path'] = config['task']['model']['module_path'][:-3]
        if 'd_feat' in config['task']['model']['kwargs']:
            config['task']['model']['kwargs']['d_feat'] = 79
        else:
            config['task']['model'].setdefault('try_kwargs', {}).update({'d_feat': 79})
    if config['task']['dataset']['class'] == 'TSDatasetH':
        config['task']['dataset']['class'] = 'DatasetH'


    # otherwise, generating a new yaml without random seed
    with open(temp_path, "w") as fp:
        yaml.dump(config, fp)
    return temp_path


class ModelRunner:

    @staticmethod
    def _init_qlib(exp_folder_name, basepath=None):  #, provider_uri="/home/booksword/traderesearch/qlib_data/rqdata"):
        if basepath is None:
            basepath = Path(os.getcwd()).resolve()
        else:
            basepath = Path(basepath)
        # init qlib
        qlib.init(
            # provider_uri=provider_uri,
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": "file:" + str(basepath / exp_folder_name),
                    "default_exp_name": "Experiment",
                },
            }
        )

    # function to run the all the models
    @only_allow_defined_args
    def run(
        self,
        times=1,
        models=None,
        dataset="Alpha158",
        universe="",
        exclude=False,
        exp_folder_name: str = "run_all_model_records",
    ):
        """
        Please be aware that this function can only work under Linux. MacOS and Windows will be supported in the future.
        Any PR to enhance this method is highly welcomed. Besides, this script doesn't support parallel running the same model
        for multiple times, and this will be fixed in the future development.

        Parameters:
        -----------
        times : int
            determines how many times the model should be running.
        models : str or list
            determines the specific model or list of models to run or exclude.
        exclude : boolean
            determines whether the model being used is excluded or included.
        dataset : str
            determines the dataset to be used for each model.
        universe  : str
            the stock universe of the dataset.
            default "" indicates that
        qlib_uri : str
            the uri to install qlib with pip
            it could be url on the github or local path (NOTE: the local path must be an absolute path)
        exp_folder_name: str
            the name of the experiment folder

        Usage:
        -------
        Here are some use cases of the function in the bash:

        The run_all_models  will decide which config to run based no `models` `dataset`  `universe`
        Example 1):

            models="lightgbm", dataset="Alpha158", universe="" will result in running the following config
            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

            models="lightgbm", dataset="Alpha158", universe="csi500" will result in running the following config
            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_csi500.yaml

        .. code-block:: bash

            # Case 1 - run all models multiple times
            python run_all_model.py run 3

            # Case 2 - run specific models multiple times
            python run_all_model.py run 3 mlp

            # Case 3 - run specific models multiple times with specific dataset
            python run_all_model.py run 3 mlp Alpha158

            # Case 4 - run other models except those are given as arguments for multiple times
            python run_all_model.py run 3 [mlp,tft,lstm] --exclude=True

            # Case 5 - run specific models for one time
            python run_all_model.py run --models=[mlp,lightgbm]

            # Case 6 - run other models except those are given as arguments for one time
            python run_all_model.py run --models=[mlp,tft,sfm] --exclude=True

            # Case 7 - run lightgbm model on csi500.
            python run_all_model.py run 3 lightgbm Alpha158 csi500

        """
        base_folder = Path("D:\Documents\TradeResearch\Stock\qlibmodels")
        self._init_qlib(exp_folder_name, base_folder)

        # get all folders
        folders = get_all_folders(models, exclude)
        # init error messages:
        # run all the model for iterations
        for fn in folders:
            # get all files
            logger.info(f"Retrieving files in {fn} ...")
            if fn == 'TFT' or \
                fn == 'GATs': # GATs depends on lstm
                continue
            if universe == "" and fn == 'TRA':
                universe = "full"
            yaml_path = get_all_files(folders[fn], dataset, universe=universe)
            if yaml_path is None:
                sys.stderr.write(f"There is no {dataset}.yaml file in {folders[fn]}\n")
                continue
            sys.stderr.write("\n")

            temp_dir = base_folder.joinpath("temp_dir")
            if not temp_dir.exists():
                temp_dir.mkdir()

            # read yaml, remove seed kwargs of model, and then save file in the temp_dir
            yaml_path = gen_yaml_files_from_example_templates(
                yaml_path, temp_dir, provider_uri=r'D:\Documents\TradeResearch\qlib_test\rqdata',
                train=['2011-01-01', '2016-12-31'], valid=['2017-01-01', '2018-12-31'], test=['2018-12-31', '2022-10-31']
            )

            for i in range(times):
                sys.stderr.write(f"Running the model: {fn} for iteration {i+1}...\n")
                try:
                    workflow(config_path=yaml_path, experiment_name=fn, uri_folder=exp_folder_name)
                except Exception as e:
                    logger.error(f"Failed run for {fn}: {str(e)}")
                    raise
        self._collect_results(exp_folder_name, dataset, base_folder)

    @staticmethod
    def _collect_results(exp_folder_name, dataset, base_folder):
        folders = get_all_folders(exp_folder_name, dataset)
        # getting all results
        sys.stderr.write(f"Retrieving results...\n")
        results = get_all_results(folders)
        if len(results) > 0:
            # calculating the mean and std
            logger.info(f"Calculating the mean and std of results...\n")
            results = cal_mean_std(results)
            # generating md table
            logger.info(f"Generating markdown table...\n")
            gen_and_save_md_table(results, dataset, base_folder)
        print("")
        # move results folder
        shutil.move(base_folder.joinpath(exp_folder_name), base_folder.joinpath(exp_folder_name + f"_{dataset}_{datetime.now().strftime('%Y%m%d%H%M%S')}"))
        shutil.move(base_folder.joinpath("table.md"), base_folder.joinpath(f"table_{dataset}_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"))


if __name__ == "__main__":
    fire.Fire(ModelRunner)  # run all the model
