import pandas as pd

from .signal_strategy import TopkDropoutStrategy, TopkKeepnDropoutStrategy


class TopkDropoutCBStrategy(TopkDropoutStrategy):
    def _generate_buy_sell_list(self, pred_score: pd.Series, current_stock_list, trade_start_time, trade_end_time):
        pred_score, current_stock_list, removed_from_population = self.filter_instruments_by_market(
            pred_score, current_stock_list, trade_start_time, trade_end_time
        )

        pred_df = pred_score.to_frame(name="score")
        pred_df["current_hold"] = pred_df.index.isin(current_stock_list)

        pred_df["call_announced"] = (
            pred_df.apply(
                lambda x: self.trade_exchange.quote.get_data(
                    x.name, trade_start_time, trade_end_time, field="$call_announced", method="ts_data_last"
                ),
                axis=1,
            ).fillna(1)
            > 0.5
        )

        pred_df["tradestatusflag"] = (
            pred_df.apply(
                lambda x: 0
                if self.trade_exchange.is_stock_tradable(x.name, trade_start_time, trade_end_time)
                else 1
                if x.current_hold
                else -1,
                axis=1,
            )
            if self.only_tradable
            else 0
        )
        pred_df.sort_values(
            by=["tradestatusflag", "score", "current_hold"], ascending=False, inplace=True, kind="stable"
        )

        # sell all sellable holdings which annouced force redemption
        sell = pred_df[pred_df.current_hold & (pred_df.tradestatusflag == 0) & pred_df.call_announced].index

        # drop items annouced force redemption
        pred_df = pred_df[~pred_df.call_announced]

        pred_df["rank"] = (pred_df["current_hold"] | (0 == pred_df["tradestatusflag"])).cumsum()
        pred_df["cum_current_hold"] = pred_df["current_hold"].cumsum()
        # sell only contains called ones now
        additional_n_drop = max(0, self.n_drop - len(sell))

        sell = sell.union(removed_from_population)

        pred_df["keep"] = pred_df["current_hold"] & (
            (pred_df["rank"] <= self.topk)
            | (pred_df["cum_current_hold"] <= len(current_stock_list) - additional_n_drop)
        )

        num_keep = pred_df.keep.sum()

        sell = sell.union(
            pred_df[~pred_df.keep & (pred_df.tradestatusflag == 0) & pred_df.current_hold].index
        ).to_list()
        buy = (
            pred_df[~pred_df.current_hold & (pred_df.tradestatusflag == 0)].iloc[: self.topk - num_keep].index.tolist()
        )

        return buy, sell


class TopkKeepnDropoutCBStrategy(TopkKeepnDropoutStrategy):
    def _generate_buy_sell_list(self, pred_score, current_stock_list, trade_start_time, trade_end_time):
        pred_score, current_stock_list, removed_from_population = self.filter_instruments_by_market(
            pred_score, current_stock_list, trade_start_time, trade_end_time
        )

        pred_df = pred_score.sort_values(ascending=False, kind="stable").to_frame()
        pred_df["current_hold"] = pred_df.index.isin(current_stock_list)
        pred_df["cum_current_hold"] = pred_df["current_hold"].cumsum()
        pred_df["tradestatusflag"] = (
            pred_df.apply(
                lambda x: 0
                if self.trade_exchange.is_stock_tradable(x.name, trade_start_time, trade_end_time)
                else 1
                if x.current_hold
                else -1,
                axis=1,
            )
            if self.only_tradable
            else 0
        )

        # sell all sellable holdings which annouced force redemption
        sell = pred_df[pred_df.current_hold & (pred_df.tradestatusflag == 0) & pred_df.call_announced].index
        # drop items annouced force redemption
        pred_df["rank"] = (pred_df["current_hold"] | (0 == pred_df["tradestatusflag"])).cumsum()
        pred_df["cum_current_hold"] = pred_df["current_hold"].cumsum()

        additional_n_drop = max(0, self.forcedropnum - len(sell))

        sell = sell.union(removed_from_population)

        pred_df["keep"] = pred_df["current_hold"] & (
            (pred_df["rank"] <= self.keepn)
            & (
                (pred_df["cum_current_hold"] <= len(current_stock_list) - additional_n_drop)
                | (pred_df["rank"] <= self.topk)
            )
        )
        if self.only_positive_score:
            pred_df["keep"] = pred_df["keep"] & (pred_df["score"] >= 0.0)
            pred_df = pred_df[(pred_df["score"] >= 0.0) | pred_df.current_hold]

        num_keep = pred_df.keep.sum()

        sell = sell.union(
            pred_df[(~pred_df.keep) & (pred_df.tradestatusflag == 0) & pred_df.current_hold].index
        ).to_list()
        buy = (
            pred_df[~pred_df.current_hold & (pred_df.tradestatusflag == 0)].iloc[: self.topk - num_keep].index.tolist()
        )

        return buy, sell
