from typing import List, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accounting.forecast_strategies import Forecast

from accounting.Account import Account, _compute_daily_std
from utils.utils import mad


def finalize_plot(plot_func: callable):
    def plot_func_wraper(self, *args, **kwargs):
        plot_func(self, *args, **kwargs)
        plt.grid(visible=True)
        plt.legend()

    return plot_func_wraper


def plot_forecast(forecast: Forecast,
                  figure_name: str = "",
                  c='b') -> None:

    bal_col = forecast.account.balance_column_name
    mean = forecast.forecast_data.loc[:, ['date', bal_col]].groupby(by='date').mean()
    std = forecast.forecast_data.loc[:, ['date', bal_col]].groupby(by='date').std()
    plt.figure(num=figure_name)
    plt.plot(mean,
             label="",
             linestyle='--',
             c=c)
    plt.fill_between(x=mean.index,
                     y1=(mean-std)[bal_col],
                     y2=(mean+std)[bal_col],
                     alpha=0.3,
                     color=c)


class AccountsList:

    def __init__(self, accounts: List[Account]):
        self.accounts = dict()
        self.__acclist__ = list()
        for account in accounts:
            self.accounts[account.name] = account
            self.__acclist__.append(account)

        self.color = plt.cm.get_cmap('tab20')(len(accounts) + 1 % len(accounts))

    def __getitem__(self, ix: int | str) -> Account:
        if isinstance(ix, int):
            item = self.__acclist__[ix]
        elif isinstance(ix, str):
            item = self.accounts[ix]
        else:
            raise ValueError(f"ix must be either int or str!")
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
            ix = total.index < pd.to_datetime(acc.most_recent_date)
            total.loc[ix, col] = total.loc[ix, col].ffill()

            if acc.status == "OPEN":
                total[col].ffill(inplace=True)

        total['balance_total'] = total.sum(axis=1)

        return total

    def interactive_codes_update(self):
        for acc in self:
            print(acc)
            acc.interactive_codes_update()

    @property
    def balance_column(self) -> pd.DataFrame:
        return self._sum_balances_columns()

    def get_balance_columns_sum(self, accounts: List[Account]) -> pd.DataFrame:
        return self._sum_balances_columns(accounts)

    def period_stats(self, date: str, end_date: str = "") -> pd.DataFrame:
        stats_list = list()
        days = 1
        for account in self:
            data, days = account.get_period_data(start_date=date, end_date=end_date)
            if data is not None:
                data['merged'] = (data.loc[:, account.numerical_names] * account.numerical_signs).sum(axis=1)
                data = data.loc[:, ['merged', 'code']]
                stats_list.append(data)

        stats = pd.concat(stats_list, axis=0)

        t_counts = stats.groupby(by='code').count().merged
        t_daily_prob = (t_counts / days)

        t_sum = stats.groupby(by='code').sum(numeric_only=True).merged
        t_ave = stats.groupby(by='code').mean(numeric_only=True).merged
        t_daily_ave = t_sum / days
        t_med = stats.groupby(by='code').median(numeric_only=True).merged

        t_daily_std = _compute_daily_std(daily_ave=t_daily_ave, period_data=stats, delta_days=days)

        t_mad = stats.groupby(by='code')['merged'].apply(mad)
        t_std = stats.groupby(by='code').std(numeric_only=True).merged

        stats = pd.DataFrame({'sums': t_sum,
                              'daily_mean': t_daily_ave,
                              'mean': t_ave,
                              'transac_median': t_med,
                              'daily_std': t_daily_std,
                              'mad': t_mad,
                              'std': t_std,
                              'daily_prob': t_daily_prob,
                              'count': t_counts},
                             index=t_std.index).replace(np.nan, 0)

        return stats

    @finalize_plot
    def plot(self, fig_name: str = "") -> None:

        if fig_name == "":
            fig_name = "Accounts plot"

        for account in self:
            account.plot(figure_name=fig_name)
        total = self.balance_column

        plt.plot(total['balance_total'], label="total balance")

    @finalize_plot
    def plot_cumulative_balances(self, accounts: List[Account], fig_name: str = "") -> None:

        if fig_name == "":
            fig_name = "Accounts plot"

        total = self.get_balance_columns_sum(accounts)

        label = "summed balance"
        for account in accounts:
            label += f" {account.name}\n"

        plt.figure(num=fig_name)
        plt.plot(total['balance_total'], label=label, c=self.color)

    @finalize_plot
    def plot_forecasts(self,
                       forecasts: Dict[str, Forecast],
                       show_total: bool = True,
                       total_offset: float = 0.,
                       figure_name: str = "") -> None:

        mean_bal = list()
        std_bal = list()

        for account_name, forecast in forecasts.items():
            forecast_data = forecast.forecast_data
            bal_col = self[account_name].balance_column_name
            mean_bal.append(forecast_data.loc[:, ['date', bal_col]].groupby(by='date').mean())
            std_bal.append(forecast_data.loc[:, ['date', bal_col]].groupby(by='date').std())
            plot_forecast(forecast=forecast, figure_name=figure_name, c=self.accounts[account_name].color)

        if show_total:
            total = pd.concat(mean_bal, axis=1)
            total = total.ffill().dropna()
            total_std = pd.concat(std_bal, axis=1)
            total_std = total_std.ffill().dropna()
            total['balance_total'] = total.sum(axis=1)
            total['balance_total'] += total_offset
            total_std['balance_std'] = total_std.std(axis=1)
            plt.plot(total['balance_total'], linestyle="--", label="", c=self.color)
            plt.fill_between(x=total.index,
                             y1=total['balance_total']-total_std['balance_std'],
                             y2=total['balance_total']+total_std['balance_std'],
                             color=self.color,
                             alpha=0.5)

    def barplot(self,
                start_date: str,
                end_date: str = "",
                excluded_codes: Sequence[str] = ('internal_cashflow', 'credit')):
        data_df = list()
        overall_len = 0
        for acc in self:
            data, period_length = acc.get_period_data(start_date=start_date, end_date=end_date)
            if data is not None:
                overall_len = max(overall_len, period_length)
                data: pd.DataFrame = data.groupby(by='code').sum(numeric_only=True)
                data['total'] = (data.loc[:, acc.numerical_names] * acc.numerical_signs).sum(axis=1)
                data.sort_values(by='total', ascending=False, inplace=True)

                data_df.append(data)

        if len(data_df) > 0:
            data = pd.concat(data_df, axis=1).loc[:, ['total']].sum(axis=1)
            data.sort_values(inplace=True, ascending=False)

            if len(excluded_codes) > 0:
                for excluded_code in excluded_codes:
                    if excluded_code in data.index.values:
                        data.drop(labels=excluded_code, inplace=True)

            income = data[data > 0].sum()
            expenses = data[data <= 0].sum()

            plt.figure(num=f"All accounts barplot - {start_date}")
            plt.title(label=f"All accounts barplot - {start_date}\n"
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

    def filter_by_code(self, code: str, start_date: str = "", end_date: str = "") -> Dict[str, pd.DataFrame]:
        filtered_data = dict()
        for account in self:
            data = account.filter_by_code(code=code, start_date=start_date, end_date=end_date)
            if data is not None:
                filtered_data[account.name] = data
        return filtered_data

    def export(self) -> None:
        for account in self:
            account.export()
