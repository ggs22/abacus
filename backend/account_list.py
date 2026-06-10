from typing import List, Dict, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.forecast_strategies import Forecast
from backend.account import Account, _compute_daily_std
from backend.plotting import (plot_accounts_list, plot_cumulative_balances,
                                 plot_forecasts, barplot)
from utils.utils import mad

# Muted qualitative palette — lower saturation than Plotly's Dark24 defaults,
# readable on both light and dark backgrounds.
_COLORS = [
    "#5b8db8",  # steel blue
    "#c4714f",  # terra cotta
    "#4a9e7a",  # sage green
    "#9068b0",  # muted violet
    "#b89440",  # muted gold
    "#4a94a8",  # teal
    "#b05878",  # muted rose
    "#6a8840",  # olive
    "#5860a0",  # indigo
    "#a06848",  # sienna
    "#4a8870",  # jade
    "#887060",  # taupe
]


class AccountsList:

    def __init__(self, accounts: List[Account]):
        self.accounts = dict()
        self.__acclist__ = list()
        for account in accounts:
            self.accounts[account.name] = account
            self.__acclist__.append(account)

        self.color = _COLORS[(len(accounts) + 1) % len(_COLORS)]

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
                total[col] = total[col].ffill()

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

    def plot(self, title: str = "Accounts") -> go.Figure:
        return plot_accounts_list(self, title=title)

    def plot_cumulative_balances(self, accounts: List[Account],
                                 title: str = "") -> go.Figure:
        return plot_cumulative_balances(self, accounts=accounts, title=title)

    def plot_forecasts(self,
                       forecasts: Dict[str, Forecast],
                       total_offset: float = 0.,
                       show_history: bool = True,
                       sum_accounts: list[str] | None = None,
                       title: str = "") -> go.Figure:
        return plot_forecasts(self, forecasts=forecasts,
                              total_offset=total_offset, show_history=show_history,
                              sum_accounts=sum_accounts, title=title)

    def barplot(self,
                start_date: str,
                end_date: str = "",
                excluded_codes: Sequence[str] = ('internal_cashflow', 'credit', 'expense_account'),
                title: str = "") -> go.Figure:
        return barplot(self, start_date=start_date, end_date=end_date,
                       excluded_codes=excluded_codes, title=title)

    def filter_by_category(self, category: str) -> "AccountsList":
        return AccountsList([acc for acc in self if acc.category == category])

    def filter_by_family(self, family: str) -> "AccountsList":
        return AccountsList([acc for acc in self if acc.family == family])

    def filter_by_institution(self, institution: str) -> "AccountsList":
        return AccountsList([acc for acc in self if acc.institution == institution])

    def filter_by_code(self, code: str, start_date: str = "", end_date: str = "") -> Dict[str, pd.DataFrame]:
        filtered_data = dict()
        for account in self:
            data = account.filter_by_code(code=code, start_date=start_date, end_date=end_date)
            if data is not None:
                filtered_data[account.name] = data
        return filtered_data

    def filter_by_description(self, description: str, start_date: str = "", end_date: str = "") -> Dict[str, pd.DataFrame]:
        filtered_data = dict()
        for account in self:
            data = account.filter_by_description(description=description, start_date=start_date, end_date=end_date)
            if data is not None:
                filtered_data[account.name] = data
        return filtered_data

    def export(self) -> None:
        for account in self:
            account.export()

    def interactive_codes_update(self):
        for account in self:
            print(account)
            account.interactive_codes_update()

    def reload(self, new_accounts: "AccountsList") -> None:
        self.accounts = new_accounts.accounts
        self.__acclist__ = new_accounts.__acclist__
        self.color = new_accounts.color

    def save(self) -> None:
        for account in self:
            account.save()

    def get_planned_transactions(self, start_date: str = "", predicted_days: int = 365) -> Dict[str, pd.DataFrame]:
        planned_transactions = dict()
        for account in self:
            data = account.get_planned_transactions(start_date, predicted_days)
            if data is not None:
                planned_transactions[account.name] = data
        return planned_transactions
