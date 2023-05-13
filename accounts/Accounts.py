import json
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import colorama
import os
import datetime
import seaborn as sns

from dataclasses import dataclass
from enum import Enum
from utils.utils import pickle_dir, months_map, data_dir


class AccountNames(Enum):
    DESJARDINS_OP = 1
    DESJARDINS_MC = 2
    DESJARDINS_PR = 3
    VISA_PP = 4
    PAYPAL = 5
    CAPITAL_ONE = 6
    CIBC = 6


class AccountStatus(Enum):
    OPEN = 1
    CLOSED = 2


class AccountType(Enum):
    OPERATIONS = 1
    CREDIT = 2
    PREPAID = 3


@dataclass
class AccountMetadata:
    raw_files_dir: str
    serialized_object_path: str
    planned_transactions_path: str
    interest_rate: float
    name: AccountNames
    columns_names: list
    assignation_file_path: str
    type: AccountType
    status: AccountStatus


def _barplot_dataframe(d: pd.DataFrame, title: str, d_avg: pd.DataFrame = None, figsize=(7, 7), show=False):
    d = d.sort_values(by='total')
    d_avg = d_avg.sort_values(by='total')
    fig = plt.figure(figsize=figsize, num=title)
    clrs = ['green' if (x > 0) else 'red' for x in d['total']]
    if d_avg is not None:
        tick_labels = [f"{val:.2f} ({val_avg:.2f})" for val, val_avg in zip(d['total'].values, d_avg['total'].values)]
    else:
        tick_labels = str(d['total'].values)
    g = sns.barplot(x='total', y=d.index, data=d, palette=clrs, tick_label=tick_labels)
    for i, (t, lbl) in enumerate(zip(d['total'], tick_labels)):
        g.text(x=(t*(t > 0) + 2), y=i, s=lbl)

    plt.title(title)
    plt.subplots_adjust(left=0.25)
    if show:
        plt.show()
    return fig


def print_codes_menu(codes, transaction):
    """
    Prints a CLI menu for manual transaction code assignation
    :param codes: List of possible transaction codes
    :param transaction: Transaction for wich a code assignation is needed
    """
    print(f'Choose transaction code for:\n'
          f'{colorama.Fore.YELLOW} {transaction} {colorama.Fore.RESET}\n'
          f'(enter corresponding number or "na"):')
    for index, code in enumerate(codes):
        print(f'{index + 1}- {code}', end=(' ' * (25 - len(f'{index + 1}- {code}'))))
        if (index + 1) % 3 == 0:
            print()
    print()


def get_common_codes(cashflow: pd.DataFrame, description_column="description") -> pd.Series:
    """
    This function returns a vector corresponding to all transaction code associated to the description vector given
    as argument

    args:
        - description   A vector of transaction description (often reffered to as "object, item, description, etc."
                        in bank statements). The relations between the descriptions and the codes are contained in
                        "assignations.csv"
    """

    descriptions = cashflow.loc[:, description_column]

    common_assignations_path = os.path.join(data_dir, 'assignation.json')
    if not os.path.exists(common_assignations_path):
        raise IOError(f"Assignation file not found at {os.path.join(data_dir, 'assignation.json')}")
    with open(os.path.join(data_dir, 'assignation.json'), 'r') as f:
        assignations = json.load(f)

    codes = list()
    for index, description in enumerate(descriptions):
        codes.append("na")
        for code, assignations_list in assignations.items():
            for assignation in assignations_list:
                if description.lower().find(assignation.lower()) != -1:
                    codes[len(codes) - 1] = code
                    break

    return pd.Series(codes)


def _is_monthly_transaction(year: int, month: int, day: int, date: datetime.date):
    return year == month == 0 and day == date.day


def _is_yearly_transaction(year: int, month: int, day: int, date: datetime.date):
    return year == 0 and month == date.month and day == date.day


def _is_periodic(year: int, month: int, day: int, date: datetime.date):
    p = _is_monthly_transaction(year=year, month=month, day=day, date=date) or \
        _is_yearly_transaction(year=year, month=month, day=day, date=date)
    return p


def _is_unique_transaction(year: int, month: int, day: int, date: datetime.date):
    return year == date.year and month == date.month and day == date.day


def _is_planned_transaction(year: int, month: int, day: int, date: datetime.date):
    return _is_periodic(year=year, month=month, day=day, date=date) or \
           _is_unique_transaction(year=year, month=month, day=day, date=date)


class Accounts:
    def __init__(self, l_accounts_list: list):
        self.accounts_list = l_accounts_list
        self.accounts_dict = dict()
        for acc in self.accounts_list:
            self.accounts_dict[acc.metadata.name.name] = acc

    def __iter__(self):
        yield from self.accounts_list

    def __len__(self):
        return len(self.accounts_list)

    def get_summed_data(self, year=None, month=None, day=None):
        sum_l = list()
        d = None
        for acc in self.accounts_list:
            t = acc.get_summed_data(year=year, month=month, day=day)
            if t.shape[0] > 0:
                sum_l.append(t)
        if len(sum_l) > 0:
            d = pd.concat(sum_l)
            d = d.groupby(by='code').sum()
        return d

    def get_summed_data_date_range(self, start_date: datetime.date, end_date: datetime.date):
        sum_l = list()
        d = None
        for acc in self.accounts_list:
            t = acc.get_summed_date_range_data(start_date=start_date, end_date=end_date)
            if t is not None:
                sum_l.append(t)
        if len(sum_l) > 0:
            d = pd.concat(sum_l)
            d = d.groupby(by='code').sum()
        return d

    def get_daily_average(self, year=None, month=None, day=None):
        av_l = list()
        d = None
        for acc in self.accounts_list:
            t = acc.get_daily_average(year=year, month=month, day=day)
            if t is not None:
                av_l.append(t)
        if len(av_l) > 0:
            d = pd.concat(av_l)
            d = d.groupby(by='code').sum()
        return d

    def get_data_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        av_l = list()
        std_l = list()
        frq_l = list()
        mean = None
        std_dev = None
        freqs = None
        for acc in self.accounts_list:
            daily_means, range_std, freqs = acc.get_date_range_daily_average(start_date=start_date, end_date=end_date)
            if daily_means is not None:
                av_l.append(daily_means)
                std_l.append(range_std)
                frq_l.append(freqs)
        if len(av_l) > 0:
            mean = pd.concat(av_l)
            mean = mean.groupby(by='code').mean()
            std_dev = pd.concat(std_l)
            std_dev = std_dev.groupby(by='code').mean()
            freqs = pd.concat(frq_l)
            freqs = freqs.groupby(by='code').mean()

        if mean is not None:
            if 'transaction_num' in mean.columns:
                mean.drop(columns=['transaction_num'], inplace=True)
            if 'internal_cashflow' in mean.index:
                mean.drop(labels=['internal_cashflow'], inplace=True)

        return mean, std_dev, freqs

    def barplot(self, year=None, month=None, day=None, show=False):

        d = self.get_summed_data(year=year, month=month, day=day)
        d_avg = self.get_daily_average(year=year, month=month, day=day)

        if d is not None and d.shape[0] > 0:
            expenses = d.loc[d['total'] < 0, 'total'].sum()
            income = d.loc[d['total'] >= 0, 'total'].sum()
            expenses_avg = d_avg.loc[d['total'] < 0, 'total'].sum()
            income_avg = d_avg.loc[d['total'] >= 0, 'total'].sum()
            title = f'{year} ' * (year is not None) + \
                    f'{month} ' * (month is not None) + \
                    f'{day} ' * (day is not None) + f'All accounts\n' + \
                    f'Expenses: {expenses:.2f}({expenses_avg:.2f})$, ' \
                    f'Income: {income:.2f}({income_avg:.2f})$, ' \
                    f'Total: {income+expenses:.2f}({income_avg+expenses_avg:.2f})$'

            return _barplot_dataframe(d=d, title=title, d_avg=d_avg, show=show)

    def barplot_date_range(self, start_date: datetime.date, end_date: datetime.date, show=False):

        d = self.get_summed_data_date_range(start_date=start_date, end_date=end_date)
        d_avg, _, _ = self.get_data_range_daily_average(start_date=start_date, end_date=end_date)

        expenses = d.loc[d['total'] < 0, 'total'].sum()
        income = d.loc[d['total'] >= 0, 'total'].sum()
        if d is not None and d.shape[0] > 0:
            title = f'{start_date} to  {end_date} All accounts\n' + \
                    f'Expenses: {expenses:.2f}$, Income: {income:.2f}$, Total: {income+expenses:.2f}$'
            return _barplot_dataframe(d=d, title=title, d_avg=d_avg, show=show)

    def get_names(self):
        name_list = list()
        for acc in self.accounts_list:
            name_list.append(acc.metadata.name.name)
        return name_list

    def get_account(self, account_name: AccountNames):
        return self.accounts_dict[account_name]

    def get_by_name(self, name: AccountNames):
        res = None
        for acc in self.accounts_list:
            if acc.metadata.name.name == name.name:
                res = acc
                break
        return res

    def get_most_recent_transaction_date(self):
        max_date = datetime.date(year=2000, month=1, day=1)
        for account in self:
            if max_date < account.most_recent_date.array.date[0]:
                max_date = account.most_recent_date.array.date[0]
        return max_date

    def get_yearly_summary(self, year: int = None):
        most_recent_transaction_date = self.get_most_recent_transaction_date()
        year = most_recent_transaction_date.year if year is None else year
        df_ = pd.DataFrame()
        end_month = 12 if year < datetime.datetime.today().year else most_recent_transaction_date.month
        for month in range(1, end_month + 1):
            summed_data = self.get_summed_data(year=year, month=month)
            if summed_data is not None:
                df_ = df_.join(other=summed_data, how='outer')
                df_.rename(columns={'total': f'{months_map[month][0]}'}, inplace=True)

        df_.replace(np.nan, 0, inplace=True)
        return df_

    def plot_yearly_summary(self, year: int = None, columns: list = None):
        year = self.get_most_recent_transaction_date().year if year is None else year
        data = self.get_yearly_summary(year=year).T
        data['expenses'] = data.loc[:, data.columns[data.columns != 'pay']].sum(axis=1)
        data['income'] = data[data > 0].sum(axis=1)
        data['total'] = data.sum(axis=1)
        data = data.loc[:, [*columns, 'expenses', 'income', 'total']] if columns is not None else data
        line_styles = ['-', '--', '-.', ':']
        plt.figure()
        for ix, (name, col) in enumerate(data.items()):
            sel = ix % len(line_styles)
            style = line_styles[sel]
            color = (sel * 0.2 % 1.0 + 0.2, random.random(), random.random())
            # plt.plot(col, c=color, linestyle=style, label=name)
            plt.scatter(col, c=color, label=name)
        # sns.relplot(data=data, kind='line', palette='muted')
        plt.legend()
        plt.xticks(rotation=90)
        plt.show()