# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from functools import partial

import plotly.graph_objs as go

import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats

from ..graph import ScatterGraph, SubplotsGraph, BarGraph, HeatmapGraph, BoxGraph, TableGraph
from ..utils import guess_plotly_rangebreaks


def _group_return(pred_label: pd.DataFrame = None, reverse: bool = False, N: int = 5, **kwargs) -> tuple:
    """

    :param pred_label:
    :param reverse:
    :param N:
    :return:
    """
    if reverse:
        pred_label["score"] *= -1

    # Groups only consider the dropna values
    pred_label_drop = pred_label.dropna(subset=["score"])

    # assign group
    def get_rank_cut(total, num_groups):
        q, r = divmod(total, num_groups)
        num_in_each_group = np.ones(num_groups, dtype=int) * q
        if r > 0:
            wing, r = divmod(r, 2)
            num_in_each_group[:wing] += 1
            num_in_each_group[-wing:] += 1
            num_in_each_group[num_groups // 2] += r
        return np.cumsum(num_in_each_group)

    ref_reverse = kwargs.get("ref_reverse", False)

    g_datetime = pred_label_drop.groupby(level="datetime")
    if "ref" in pred_label.columns:

        def _groupify(pred_label_day: pd.DataFrame, num_groups: int, reverse=False, ref_reverse=False):
            pred_label_day = pred_label_day.droplevel(level="datetime").sort_values(
                by=["score", "ref"], ascending=(reverse, ref_reverse)
            )
            group_rank_bound = get_rank_cut(len(pred_label_day), num_groups)
            return pd.Series(
                [f"Group{gidx+1}" for gidx in np.searchsorted(group_rank_bound, np.arange(1, len(pred_label_day) + 1))],
                index=pred_label_day.index,
            )

        pred_label_drop["group"] = g_datetime.apply(
            partial(_groupify, num_groups=N, reverse=reverse, ref_reverse=ref_reverse)
        )

    else:

        def _groupify(pred_label_day: pd.DataFrame, num_groups: int, reverse=False):
            ranks = (
                ((2 * float(reverse) - 1) * pred_label_day.droplevel(level="datetime")["score"])
                .rank(method="dense")
                .astype(int)
            )
            group_rank_bound = get_rank_cut(ranks.max(), num_groups)
            return pd.Series(
                [f"Group{gidx+1}" for gidx in np.searchsorted(group_rank_bound, ranks)], index=pred_label_day.index
            )

        pred_label_drop["group"] = g_datetime.apply(partial(_groupify, num_groups=N, reverse=reverse))

    t_df = pred_label_drop.pivot_table(values="label", index="datetime", columns="group", aggfunc=np.mean)

    # Long-benchmark
    benchmark = kwargs.get("benchmark", "mean")
    if isinstance(benchmark, str):
        benchmark_name = benchmark
        benchmark = getattr(g_datetime["label"], benchmark_name)()
    elif isinstance(benchmark, pd.Series):
        benchmark_name = benchmark.name if bool(benchmark.name) else "benchmark"
        benchmark = benchmark.reindex(t_df.index)
    else:
        raise TypeError(f"Invalid benchmark type: {type(benchmark)}")
    t_df[f"long-{benchmark_name}"] = t_df["Group1"] - benchmark

    # Long-Short
    t_df["long-short"] = t_df["Group1"] - t_df["Group%d" % N]

    t_df = t_df.dropna(how="all")  # for days which does not contain label
    # Cumulative Return By Group
    group_scatter_figure = ScatterGraph(
        t_df.cumsum(),
        layout=dict(
            title="Cumulative Return",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(t_df.index))),
        ),
    ).figure

    # statistics of returns of groups and long-short and long-bench
    def _cal_statics(return_series):
        n = len(return_series)
        mean_ret = return_series.mean()
        std_ret = return_series.std()
        cum_ret = return_series.sort_index().cumsum()
        max_dd = (cum_ret.cummax() - cum_ret).max()
        years = (cum_ret.index[-1] - cum_ret.index[0]) / pd.Timedelta(days=365)

        return pd.Series(
            [
                mean_ret,
                std_ret,
                return_series.skew(),
                return_series.kurt(),
                cum_ret[-1] / years,
                mean_ret / std_ret * np.sqrt(n),
                max_dd,
                cum_ret[-1] / max_dd,
            ],
            index=[
                "periodwise return mean",
                "return std",
                "return skew",
                "return kurt",
                "annual return",
                "Sharp ratio",
                "max drawdown",
                "Calmar ratio",
            ],
        )

    stats_df = t_df.agg(_cal_statics)
    stats_table_figure = TableGraph(
        stats_df,
        graph_kwargs=dict(
            cell_kwargs=dict(
                format=[[None] * 8]
                + [[".4%", ".4%", ".4f", ".4f", ".4%", ".4f", ".4%", ".4f"]] * (len(stats_df.columns))
            )
        ),
        layout=dict(title="Group Return Summary"),
    ).figure

    t_df = t_df.loc[:, ["long-short", f"long-{benchmark_name}"]]

    pred_label_drop["excess"] = pred_label_drop["label"] - benchmark
    box_figure = BoxGraph(pred_label_drop, data_column="excess", category_column="group").figure

    _bin_size = float((t_df.std() / 5).min())
    group_hist_figure = SubplotsGraph(
        t_df,
        # kind_map=dict(kind="DistplotGraph", kwargs=dict(bin_size=_bin_size)),
        sub_graph_data=[
            (box_figure, dict(row=1, col=3)),
            ("long-short", dict(row=1, col=1, kind="DistplotGraph", graph_kwargs=dict(bin_size=_bin_size))),
            (f"long-{benchmark_name}", dict(row=1, col=2, kind="DistplotGraph", graph_kwargs=dict(bin_size=_bin_size))),
        ],
        subplots_kwargs=dict(
            rows=1,
            cols=3,
            print_grid=False,
            subplot_titles=["long-short", f"long-{benchmark_name}", "group box plot"],
        ),
    ).figure

    return group_scatter_figure, stats_table_figure, group_hist_figure


def _plot_qq(data: pd.Series = None, dist=stats.norm) -> go.Figure:
    """

    :param data:
    :param dist:
    :return:
    """
    # NOTE: plotly.tools.mpl_to_plotly not actively maintained, resulting in errors in the new version of matplotlib,
    # ref: https://github.com/plotly/plotly.py/issues/2913#issuecomment-730071567
    # removing plotly.tools.mpl_to_plotly for greater compatibility with matplotlib versions
    _plt_fig = sm.qqplot(data.dropna(), dist=dist, fit=True, line="45")
    plt.close(_plt_fig)
    qqplot_data = _plt_fig.gca().lines
    fig = go.Figure()

    fig.add_trace(
        {
            "type": "scatter",
            "x": qqplot_data[0].get_xdata(),
            "y": qqplot_data[0].get_ydata(),
            "mode": "markers",
            "marker": {"color": "#19d3f3"},
        }
    )

    fig.add_trace(
        {
            "type": "scatter",
            "x": qqplot_data[1].get_xdata(),
            "y": qqplot_data[1].get_ydata(),
            "mode": "lines",
            "line": {"color": "#636efa"},
        }
    )
    del qqplot_data
    return fig


def _pred_ic(pred_label: pd.DataFrame = None, rank: bool = False, **kwargs) -> tuple:
    """

    :param pred_label:
    :param rank:
    :return:
    """
    if rank:
        ic = pred_label.groupby(level="datetime").apply(
            lambda x: x["label"].rank(pct=True).corr(x["score"].rank(pct=True))
        )
    else:
        ic = pred_label.groupby(level="datetime").apply(lambda x: x["label"].corr(x["score"]))

    _index = ic.index.get_level_values(0).astype("str").str.replace("-", "").str.slice(0, 6)
    _monthly_ic = ic.groupby(_index).mean()
    _monthly_ic.index = pd.MultiIndex.from_arrays(
        [_monthly_ic.index.str.slice(0, 4), _monthly_ic.index.str.slice(4, 6)],
        names=["year", "month"],
    )

    # fill month
    _month_list = pd.date_range(
        start=pd.Timestamp(f"{_index.min()[:4]}0101"),
        end=pd.Timestamp(f"{_index.max()[:4]}1231"),
        freq="1M",
    )
    _years = []
    _month = []
    for _date in _month_list:
        _date = _date.strftime("%Y%m%d")
        _years.append(_date[:4])
        _month.append(_date[4:6])

    fill_index = pd.MultiIndex.from_arrays([_years, _month], names=["year", "month"])

    _monthly_ic = _monthly_ic.reindex(fill_index)

    _ic_df = ic.to_frame("ic")
    ic_bar_figure = ic_figure(_ic_df, kwargs.get("show_nature_day", True))

    ic_heatmap_figure = HeatmapGraph(
        _monthly_ic.unstack(),
        layout=dict(title="Monthly IC", yaxis=dict(tickformat=",d")),
        graph_kwargs=dict(xtype="array", ytype="array"),
    ).figure

    def _cal_statistic_ic(s):
        mean = s.mean()
        std = s.std()
        skew = s.skew()
        kurt = s.kurt()
        t_stat, p_value = stats.ttest_1samp(s, 0)
        return pd.Series(
            [mean, std, mean / std, t_stat, p_value, skew, kurt],
            index=["IC mean", "IC std", "Risk-adjusted IC", "t-stat(IC)", "p-value(IC)", "IC skew", "IC kurtosis"],
        )

    _stats_ic_df = _ic_df.agg(_cal_statistic_ic).rename(columns={"ic": "stats."})
    _stats_ic_table_figure = TableGraph(_stats_ic_df, graph_kwargs=dict(cell_kwargs=dict(format=[None, ".4f"]))).figure

    dist = stats.norm
    _qqplot_fig = _plot_qq(ic, dist)

    if isinstance(dist, stats.norm.__class__):
        dist_name = "Normal"
    else:
        dist_name = "Unknown"

    _bin_size = ((_ic_df.max() - _ic_df.min()) / 20).min()
    _sub_graph_data = [
        (_stats_ic_table_figure, dict(row=1, col=1)),
        (
            "ic",
            dict(
                row=1,
                col=2,
                name="",
                kind="DistplotGraph",
                graph_kwargs=dict(bin_size=_bin_size),
            ),
        ),
        (_qqplot_fig, dict(row=1, col=3)),
    ]
    ic_hist_figure = SubplotsGraph(
        _ic_df.dropna(),
        kind_map=dict(kind="HistogramGraph", kwargs=dict()),
        subplots_kwargs=dict(
            rows=1,
            cols=3,
            print_grid=False,
            subplot_titles=["IC Stats.", "IC Hist", "IC %s Dist. Q-Q" % dist_name],
            specs=[[{"type": "table"}, {"type": "scatter"}, {"type": "scatter"}]],
        ),
        sub_graph_data=_sub_graph_data,
        layout=dict(
            yaxis2=dict(title="Observed Quantile"),
            xaxis2=dict(title=f"{dist_name} Distribution Quantile"),
            height=400,
        ),
    ).figure

    return ic_bar_figure, ic_heatmap_figure, ic_hist_figure


def _pred_autocorr(pred_label: pd.DataFrame, lag=1, **kwargs) -> tuple:
    pred = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument")["score"].shift(lag)
    ac = pred.groupby(level="datetime").apply(lambda x: x["score"].rank(pct=True).corr(x["score_last"].rank(pct=True)))
    _df = ac.to_frame("value")
    # _df.index = _df.index.strftime("%Y-%m-%d")

    ac_figure = ScatterGraph(
        _df,
        layout=dict(
            title="Auto Correlation",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(_df.index))),
        ),
    ).figure
    return ac_figure,


def _pred_turnover(pred_label: pd.DataFrame, N=5, lag=1, **kwargs) -> tuple:
    pred = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument")["score"].shift(lag)
    top = pred.groupby(level="datetime").apply(
        lambda x: 1
        - x.nlargest(len(x) // N, columns="score").index.isin(x.nlargest(len(x) // N, columns="score_last").index).sum()
        / (len(x) // N)
    )
    bottom = pred.groupby(level="datetime").apply(
        lambda x: 1
        - x.nsmallest(len(x) // N, columns="score")
        .index.isin(x.nsmallest(len(x) // N, columns="score_last").index)
        .sum()
        / (len(x) // N)
    )
    r_df = pd.DataFrame(
        {
            "Top": top,
            "Bottom": bottom,
        }
    )
    turnover_figure = ScatterGraph(
        r_df,
        layout=dict(
            title="Top-Bottom Turnover",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(r_df.index))),
        ),
    ).figure
    return turnover_figure,


def ic_figure(ic_df: pd.DataFrame, show_nature_day=True, **kwargs) -> go.Figure:
    """IC figure

    :param ic_df: ic DataFrame
    :param show_nature_day: whether to display the abscissa of non-trading day
    :param **kwargs: contains some parameters to control plot style in plotly. Currently, supports
       - `rangebreaks`: https://plotly.com/python/time-series/#Hiding-Weekends-and-Holidays
    :return: plotly.graph_objs.Figure
    """
    if show_nature_day:
        date_index = pd.date_range(ic_df.index.min(), ic_df.index.max())
        ic_df = ic_df.reindex(date_index)

    ic_bar_figure = BarGraph(
        ic_df,
        layout=dict(
            title="Information Coefficient (IC)",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(ic_df.index))),
        ),
    ).figure
    return ic_bar_figure


def factor_performance_graph(
    pred_label: pd.DataFrame,
    lag: int = 1,
    N: int = 5,
    reverse=False,
    rank=False,
    graph_names: list = ["group_return", "pred_ic", "pred_autocorr", "pred_turnover"],
    show_notebook: bool = True,
    show_nature_day=True,
    **kwargs,
) -> [list, tuple]:
    """Factor performance

    :param pred_label: index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[score,
    label]**. It is usually same as the label of model training(e.g. "Ref($close, -2)/Ref($close, -1) - 1").


            .. code-block:: python

                instrument  datetime        score       label
                SH600004    2017-12-11  -0.013502       -0.013502
                                2017-12-12  -0.072367       -0.072367
                                2017-12-13  -0.068605       -0.068605
                                2017-12-14  0.012440        0.012440
                                2017-12-15  -0.102778       -0.102778


    :param lag: `pred.groupby(level='instrument')['score'].shift(lag)`. It will be only used in the auto-correlation computing.
    :param N: group number, default 5.
    :param reverse: if `True`, `pred['score'] *= -1`.
    :param rank: if **True**, calculate rank ic.
    :param graph_names: graph names; default ['cumulative_return', 'pred_ic', 'pred_autocorr', 'pred_turnover'].
    :param show_notebook: whether to display graphics in notebook, the default is `True`.
    :param show_nature_day: whether to display the abscissa of non-trading day.
    :param **kwargs: contains some parameters to control plot style in plotly. Currently, supports
       - `rangebreaks`: https://plotly.com/python/time-series/#Hiding-Weekends-and-Holidays
    :return: if show_notebook is True, display in notebook; else return `plotly.graph_objs.Figure` list.
    """
    figure_list = []
    for graph_name in graph_names:
        fun_res = eval(f"_{graph_name}")(
            pred_label=pred_label, lag=lag, N=N, reverse=reverse, rank=rank, show_nature_day=show_nature_day, **kwargs
        )
        figure_list += fun_res

    if show_notebook:
        BarGraph.show_graph_in_notebook(figure_list)
    else:
        return figure_list
