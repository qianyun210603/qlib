import ruamel.yaml as yaml
import qlib
from qlib.utils import init_instance_by_config, flatten_dict, fill_placeholder
from qlib.workflow import R
from qlib.model.trainer import task_train
from qlib.workflow.cli import workflow


if __name__ == '__main__':
    config_path = "/home/booksword/traderesearch/qlib_run_all_models/20221102" \
                  "/temp_dir/workflow_config_lightgbm_Alpha158_csi300.yaml"
    with open(config_path, 'r') as f:
         config = yaml.safe_load(f)

    qlib.init(**config['qlib_init'])

    with R.start(experiment_name="workflow"):
        rec = R.get_recorder()
        task_config = config['task']
        R.log_params(**flatten_dict(task_config))

        dataset = init_instance_by_config(task_config['dataset'])

        model = init_instance_by_config(task_config['model'])
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
        task_config = fill_placeholder(task_config, placehorder_value)

        records = task_config.get("record", [])
        if isinstance(records, dict):  # prevent only one dict
            records = [records]
        for record in records:
            # Some recorder require the parameter `model` and `dataset`.
            # try to automatically pass in them to the initialization function
            # to make defining the tasking easier
            r = init_instance_by_config(
                record,
                recorder=rec,
                default_module="qlib.workflow.record_temp",
                try_kwargs={"model": model, "dataset": dataset},
            )
            r.generate()
