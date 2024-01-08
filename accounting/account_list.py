from copy import deepcopy
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import accounting.forecast_strategies as forecast_strategies

from accounting.Account import Account


def finalize_plot(plot_func: callable):
    def plot_func_wraper(self, *args, **kwargs):
        plot_func(self, *args, **kwargs)
        plt.grid(visible=True)
        plt.legend()

    return plot_func_wraper


class AccountsList:

    def __init__(self, accounts: List[Account]):
        self.accounts = dict()
        self.strategies = dict()
        self.__acclist__ = list()
        for account in accounts:
            self.accounts[account.name] = account
            self.__acclist__.append(account)
            self.strategies[account.name] = list()
            if account.conf.forecast_strategies is not None:
                self.strategies[account.name] += [
                    getattr(forecast_strategies, strategy)() for strategy in account.conf.forecast_strategies
                ]
        self.color = cm.get_cmap('tab10')(5)

    def __getitem__(self, ix) -> Account:
        if isinstance(ix, int):
            item = self.__acclist__[ix]
        elif isinstance(ix, str):
            item = self.accounts[ix]
        return item

    def _sum_balances_columns(self, accounts: List[Account] | None = None) -> pd.DataFrame:
        accounts = accounts if accounts is not None else self
        balances = list()
        for account in accounts:
            balance = account.balance_column
            balance = balance.groupby(level=0).last()
            balances.append(balance)

        total = pd.concat(balances, axis=1)

        for col in total.columns:
            acc = self[col.split('_')[1]]
            ix = total.index < acc.most_recent_date.item()
            total.loc[ix, col] = total.loc[ix, col].ffill()

            if acc.status == "OPEN":
                total[col].ffill(inplace=True)

        total['balance_total'] = total.sum(axis=1)

        return total

    @property
    def balance_column(self) -> pd.DataFrame:
        return self._sum_balances_columns()

    def get_balance_columns_sum(self, accounts: List[Account]) -> pd.DataFrame:
        return self._sum_balances_columns(accounts)

    def period_stats(self, date: str, date_end: str = "", **kwargs) -> pd.DataFrame:
        stats_list = list()
        for account in self:
            swap: bool = account.use_legacy_stats
            account.use_legacy_stats = False
            stats_list.append(account.period_stats(date=date, date_end=date_end))
            account.use_legacy_stats = swap

        stats = pd.DataFrame(columns=stats_list[0].columns)
        df1_reset = deepcopy(stats)
        df1_reset['index'] = stats.index
        for ix in range(0, len(stats_list)):
            if stats_list[ix] is not None:
                df2_reset = deepcopy(stats_list[ix])
                df2_reset['index'] = stats_list[ix].index
                stats = df1_reset.merge(df2_reset, on=['index', *stats.columns], how='outer').set_index('index')
                # TODO make sure that this reduction is valid
                stats = stats.groupby(stats.index).mean()
                df1_reset = stats

        return stats

    def get_balance_prediction(self, predicted_days: int = 365, **kwargs) -> pd.DataFrame:
        preds: List[pd.DataFrame] = list()
        for account in self:
            pred: pd.DataFrame = self.predict_strategy.predict(account=account,
                                                               predicted_days=predicted_days,
                                                               **kwargs)
            if pred is not None:
                reduced_pred = pred.loc[:, ['date', 'description', 'balance']].groupby(
                    by=['description', 'date'], group_keys=False
                ).last()
                reduced_pred[f'balance_{account.name}'] = reduced_pred['balance']
                reduced_pred.drop(columns='balance', inplace=True)
                preds.append(reduced_pred)
        pred = pd.concat(preds, axis=1).sort_index(axis=0, level=[0, 1])
        pred.ffill(inplace=True)
        pred['total_balance'] = pred.sum(axis=1, numeric_only=True)
        pred.dropna(inplace=True)
        return pred

    @finalize_plot
    def plot(self, fig_name: str = "") -> None:

        if fig_name == "":
            fig_name = "Accounts plot"

        for account in self:
            account.plot(figure_name=fig_name)
        total = self.balance_column

        plt.plot(total['balance_total'], label="total balance", c=self.color)

    @finalize_plot
    def plot_cumulative_balances(self, accounts: List[Account], fig_name: str = "") -> None:

        if fig_name == "":
            fig_name = "Accounts plot"

        total = self.get_balance_columns_sum(accounts)

        label = "summed balance"
        for account in accounts:
            label += f" {account.name}"

        plt.figure(num=fig_name)
        plt.plot(total['balance_total'], label=label)

    @finalize_plot
    def plot_predictions(self,
                         predicted_days: int = 356,
                         figure_name: str = "",
                         simulation_date: str = "",
                         **kwargs):
        mean_bal = list()
        std_bal = list()
        for account in self:
            forecast_once = False  # only one forecast will serve to total calculation
            for strategy in self.strategies[account.name]:
                strategy.plot_prediction(account=account,
                                         predicted_days=predicted_days,
                                         figure_name=figure_name,
                                         simulation_date=simulation_date,
                                         **kwargs)
                if account.prediction is not None and not forecast_once:
                    pred = account.prediction.loc[:, ['date', 'balance']].rename(columns={'balance': f'balance_{account.name}'})
                    mean_bal.append(pred.groupby(by='date').mean())
                    std_bal.append(pred.groupby(by='date').std())
                    forecast_once = True

        show_total = kwargs.pop('show_total', True)
        if show_total:
            total = pd.concat(mean_bal, axis=1)
            total = total.ffill().dropna()
            total_std = pd.concat(std_bal, axis=1)
            total_std = total_std.ffill().dropna()
            total['balance_total'] = total.sum(axis=1)
            total_std['balance_std'] = total_std.std(axis=1)
            plt.plot(total['balance_total'], linestyle="--", label="", c=self.color)
            plt.fill_between(x=total.index,
                             y1=total['balance_total']-total_std['balance_std'],
                             y2=total['balance_total']+total_std['balance_std'],
                             color=self.color,
                             alpha=0.5)

    def barplot(self, period_seed_date: str, date_end: str = ""):
        data_df = list()
        overall_len = 0
        for acc in self:
            data, period_length = acc.get_period_data(period_seed_date=period_seed_date, date_end=date_end)
            if data is not None:
                overall_len = max(overall_len, period_length)
                data: pd.DataFrame = data.groupby(by='code').sum(numeric_only=True)
                data['total'] = (data.loc[:, acc.numerical_names] * acc.numerical_signs).sum(axis=1)
                data.sort_values(by='total', ascending=False, inplace=True)

                data_df.append(data)

        data = pd.concat(data_df, axis=1).loc[:, ['total']].sum(axis=1)
        data.sort_values(inplace=True, ascending=False)

        income = data[data > 0].sum()
        expenses = data[data <= 0].sum()

        plt.figure(num=f"All accounts barplot - {period_seed_date}")
        plt.title(label=f"All accounts barplot - {period_seed_date}\n"
                        f"in: {income: .2f}, out: {expenses: .2f}, bal: {income + expenses: .2f}")

        colors = list()
        for amount in data:
            color = (.75, 0, 0) if amount <= 0 else (0, .75, 0)
            colors.append(color)
        plt.barh(y=data.index, width=data, color=colors)
        for stat in data.index:
            plt.text(x=max(0, data[stat]) + 10,
                     y=stat,
                     s=f"{data[stat]: .2f} / {data[stat] / overall_len: .2f}")

    def filter_by_code(self, code: str, period_seed_date: str = "", date_end: str = "") -> Dict[str, pd.DataFrame]:
        filtered_data = dict()
        for account in self:
            data = account.filter_by_code(code=code, period_seed_date=period_seed_date, date_end=date_end)
            if data is not None:
                filtered_data[account.name] = data
        return filtered_data

    def export(self) -> None:
        for account in self:
            account.export()
