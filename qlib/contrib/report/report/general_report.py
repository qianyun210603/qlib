from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from qlib.contrib.report import analysis_model, analysis_position
from qlib.contrib.report.utils import convert_fig_to_base64_str
from qlib.workflow import R

TEMPLATE_PATH = Path(__file__).resolve().parent.joinpath("template")


def _fill_general_templates(title, component_list, report_type="html"):
    env = Environment(loader=FileSystemLoader(TEMPLATE_PATH))
    template = env.get_template(f"general_report_template.{report_type}")
    jinjaout = template.render(title=title, component_list=component_list)
    return jinjaout


def generate_full_report_for_recorder(
    exp_id=None,
    exp_name=None,
    record_name=None,
    recorder_id=None,
    recorder=None,
    report_contents=("Backtest Performance", "Model Evaluation"),
    dynamic_figure=True,
    **kwargs,
):
    r"""

    Parameters
    ----------
    exp_id: str
    exp_name: str
    record_name: str
    recorder_id: str
    recorder: Recorder
    the recorder which contains the experiment run results, if provided, exp_id, exp_name, record_name, recorder_id
    will be ignored
    report_contents: Set
    \*\*kwargs: Other params for different report
      - benchmark: {'average'|pd.Series}
       used in `Model Evaluation`, benchmark to be compared with group returns
      - stratify_groups: int
       used in `Model Evaluation`, number of stratify groups
    """
    if recorder is None:
        assert (exp_id is not None or exp_name is not None) and (
            recorder_id is not None or record_name is not None
        ), "Please input at least one of experiment/recorder id or name before retrieving experiment/recorder."
        exp = R.get_exp(experiment_id=exp_id, experiment_name=exp_name)
        recorder = exp.get_recorder(recorder_id=recorder_id, recorder_name=record_name)
    else:
        exp = R.get_exp(experiment_id=recorder.experiment_id)
    exp_name = str(exp.name) if bool(exp.name) else f"Exp[{exp.id}]"
    record_name = str(recorder.name) if bool(recorder.name) else f"Recorder[{recorder.id}]"
    title = exp_name + " -- " + record_name

    content_list = [
        dict(
            header="Recorder Info",
            type="subsections",
            content=[
                dict(header="Run Params", type="itemlist", content=recorder.list_params()),
                dict(header="Tags", type="itemlist", content=recorder.list_tags()),
            ],
        )
    ]

    def _process_figure_list(figure_list):
        if dynamic_figure:
            return "html", "".join(fig.to_html(full_html=False) for fig in figure_list)
        return "base64imagelist", [convert_fig_to_base64_str(fig) for fig in figure_list]

    if "Backtest Performance" in report_contents:
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        fig_list = analysis_position.report_graph(report_normal_df, show_notebook=False)
        mytype, mycontent = _process_figure_list(fig_list)
        risk_fig_list = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
        risktype, riskcontent = _process_figure_list(risk_fig_list)
        content_list.append(
            dict(
                header="Backtest Performance",
                type="subsections",
                content=[
                    dict(
                        type="html",
                        content=analysis_df.risk.unstack().to_html(
                            formatters={"information_ratio": lambda x: f"{x:.4f}"}, float_format=lambda x: f"{x:.4%}"
                        ),
                    ),
                    dict(type=mytype, content=mycontent),
                    dict(header="Monthly Risk Summary", type=risktype, content=riskcontent),
                ],
            )
        )

    if "Model Evaluation" in report_contents:
        pred_df = recorder.load_object("pred.pkl")
        label_df = recorder.load_object("label.pkl")
        pred_label = pd.merge(
            label_df.iloc(axis=1)[0].rename("label"),
            pred_df.iloc(axis=1)[0].rename("score"),
            left_index=True,
            right_index=True,
            how="left",
        )
        stratify_groups = kwargs.pop("stratify_groups", 5)
        fig_list = analysis_model.model_performance_graph(pred_label, N=stratify_groups, show_notebook=False, **kwargs)
        mytype, mycontent = _process_figure_list(fig_list)
        content_list.append(
            dict(
                header="Model Evaluation",
                type=mytype,
                content=mycontent,
            )
        )

    return _fill_general_templates(title, content_list, "html")


if __name__ == "__main__":
    import qlib
    from qlib.config import C
    from qlib.constant import REG_CN
    from qlib.data import D

    exp_mgr = C["exp_manager"]
    exp_mgr["kwargs"]["uri"] = "file:" + r"D:\Documents/TradeResearch/Stock/qlibmodels/all_model_Alpha101_csi500"
    provider_uri = r"D:\Documents\TradeResearch\qlib_test\rqdata"
    qlib.init(provider_uri=provider_uri, region=REG_CN, exp_manager=exp_mgr)

    bench_price = D.features(["SH000905"], ["$close"], start_time="2018-12-20", end_time="2022-10-31", freq="day")
    bench_ret = bench_price.loc["SH000905", "$close"] / bench_price.loc["SH000905", "$close"].shift(1) - 1
    my_html = generate_full_report_for_recorder(exp_name="LightGBM", recorder_id="e0e7c34fd7284f1ea37d976d7bc514d4")
    with open(r"E:\test_qlib_perf_report.html", "w") as f:
        f.write(my_html)
