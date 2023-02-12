import json
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pickle
import colorama
import os
import datetime
import re
import seaborn as sns

from dataclasses import dataclass
from enum import Enum
from abc import abstractmethod, ABC
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from utils.utils import pickle_dir, months_map, data_dir
from tqdm import tqdm
from typing import List, Union


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
    # unique_keys: list
    type: AccountType
    status: AccountStatus


def take_docstring(fn):
    fn.__doc__ = sns.relplot.__doc__


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


class Account(ABC):
    def __init__(self, lmetadata: AccountMetadata):

        self.metadata = lmetadata
        self.transaction_data = None
        self.col_mask = None
        self._load_from_raw_files()
        if self.transaction_data is not None:
            self.most_recent_date = self.transaction_data.tail(n=1).date
        self.planned_transactions = self._load_planned_transactions()

    def __getitem__(self, item):
        ret = self.transaction_data.loc[item, :]
        return ret

    def get_data_by_date(self, year=None, month=None, day=None) -> pd.DataFrame:

        res = self.transaction_data
        if year is not None:
            res = res[res['date'].array.year == year]

        if month is not None:
            res = res[res['date'].array.month == month]

        if day is not None:
            res = res[res['date'].array.day == day]

        return res

    def get_data_by_date_range(self, start_date: datetime.date = None, end_date: datetime.date = None) -> pd.DataFrame:

        res = self.transaction_data
        res = res[(res['date'].array.date >= start_date) & (res['date'].array.date <= end_date)]

        return res

    def get_summed_data(self, year=None, month=None, day=None):
        if self.col_mask is not None:
            d = self.get_data_by_date(year=year, month=month, day=day).loc[:, self.col_mask]
        else:
            d = self.get_data_by_date(year=year, month=month, day=day)
        return d.groupby(by='code').sum(numeric_only=True)

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        if self.col_mask is not None:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date).loc[:, self.col_mask]
        else:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date)
        return d.groupby(by='code').sum(numeric_only=True)

    def get_date_range_data_std(self, start_date: datetime.date, end_date: datetime.date):
        if self.col_mask is not None:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date).loc[:, self.col_mask]
        else:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date)
        return d.groupby(by='code').std()

    def get_date_range_data_mean(self, start_date: datetime.date, end_date: datetime.date):
        if self.col_mask is not None:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date).loc[:, self.col_mask]
        else:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date)
        return d.groupby(by='code').mean()

    def get_date_range_data_code_count(self, start_date: datetime.date, end_date: datetime.date):
        if self.col_mask is not None:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date).loc[:, self.col_mask]
        else:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date)
        return d.value_counts()

    def get_daily_average(self, year=None, month=None, day=None):
        """
        :param year: year of calculated average
        :param month: month of calculated average
        :param day: day of calculated average
        :return: daily average spending per spending code
        """

        delta = 0
        if day is not None:
            delta = 1
        elif day is None and month is not None:
            if month in [1, 3, 5, 7, 8, 10, 12]:
                delta = 31
            elif month in [4, 6, 9, 11]:
                delta = 30
            elif month == 2:
                delta = 28
        elif day is None and month is None and year is not None:
            delta = 365

        d = self.get_data_by_date(year=year, month=month, day=day)
        if d.shape[0] > 0:
            summed_data = self.get_summed_data(year=year, month=month, day=day)
            res = np.divide(summed_data, delta)
        else:
            res = None

        return res

    def get_date_range_daily_average(self, start_date: datetime.date, end_date:datetime.date):
        """
        :param end_date: first date of date range
        :param start_date: last date of date range
        :return: daily average spending per spending code for a given date rage
        """
        raise NotImplementedError(f"The \"get_date_range_daily_average\" method has not been implementted for "
                                  f"the {self.__class__} class!")

    def get_data_by_code(self, code: str, year=None, month=None, day=None):
        d = self.get_data(year=year, month=month, day=day)
        return d.loc[d['code'] == code]

    def _load_from_raw_files(self, update_data=False) -> None:
        df = None
        if os.path.exists(self.metadata.serialized_object_path) and not update_data:
            with open(self.metadata.serialized_object_path, 'rb') as save_file:
                df = pickle.load(save_file)

        elif os.path.exists(self.metadata.raw_files_dir):
            if not update_data:
                cash_flow = pd.DataFrame(columns=self.metadata.columns_names)
            else:
                cash_flow = self.transaction_data
                if cash_flow is None:
                    cash_flow = pd.DataFrame(columns=self.metadata.columns_names)

            csv_files = os.listdir(self.metadata.raw_files_dir)
            for x in csv_files:
                if x[-4:] == '.csv':
                    cash_flow = cash_flow.append(
                        pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
                                    encoding='latin1',
                                    names=self.metadata.columns_names),
                        ignore_index=True)

            # Convert strings to actual date time objects
            if 'date' in cash_flow.columns:
                cash_flow['date'] = pd.to_datetime(cash_flow['date'], format='%Y-%m-%d')

            # Adds column,and inputs transaction code
            cash_flow['code'] = get_common_codes(cash_flow)
            cash_flow = cash_flow.replace(np.nan, 0)
            cash_flow.drop_duplicates(subset=cash_flow.columns[cash_flow.columns != 'code'], inplace=True)
            cash_flow.sort_values(by=['date'], ascending=True, inplace=True)
            cash_flow.reset_index(drop=True, inplace=True)

            df = cash_flow

        if df is None:
            raise FileNotFoundError(f'No serialized or raw data (in {self.metadata.raw_files_dir}) was found for '
                                    f'{self.metadata.name}')
        else:
            # TODO: try assigning all df attributs to self
            self.transaction_data = df

    def _load_planned_transactions(self) -> pd.DataFrame:
        df = None
        if os.path.exists(self.metadata.planned_transactions_path):
            df = pd.read_csv(self.metadata.planned_transactions_path)
        return df

    def get_current_balance(self) -> float:
        if 'balance' in self.transaction_data.columns:
            return self.transaction_data.tail(n=1)['balance'].values[0]
        else:
            return 0

    @staticmethod
    def _save_prompt():
        ans = None
        while ans not in ['y', 'n', '']:
            ans = input('Save new entries?([y]/n)').lower()
        return ans in ['y', '']

    def to_pickle(self):
        with open(self.metadata.serialized_object_path, 'wb') as save_file:
            pickle.dump(self.transaction_data, save_file)

    def get_data(self, year=None, month=None, day=None) -> pd.DataFrame:
        d = self.get_data_by_date(year=year, month=month, day=day)
        if self.col_mask is not None:
            return d[self.col_mask]
        else:
            return d[self.metadata.columns_names]

    def barplot(self, year=None, month=None, day=None, show=False):

        d = self.get_summed_data(year=year, month=month, day=day)
        d_avg = self.get_daily_average(year=year, month=month, day=day)

        if d is not None and d.shape[0] > 0:
            title = f'{year} ' * (year is not None) + \
                    f'{month} ' * (month is not None) + \
                    f'{day} ' * (day is not None) + f'{self.get_name()}'
            return _barplot_dataframe(d=d, title=title, d_avg=d_avg, show=show)

    def get_name(self):
        s = str(self.metadata.name).split('.')[1]
        s = s.replace('_', ' ')
        return s

    def change_transaction_code(self, ix, code):
        self.transaction_data.loc[ix, 'code'] = code

    def apply_description_filter(self, pattern: str, regex=False):
        ret = self.transaction_data.loc[
              self.transaction_data["description"].str.contains(pattern, regex=regex), :]
        return ret

    def clear_month(self, year: int = -1, month: int = -1, inplace=False):
        year = datetime.datetime.today().year * (year == -1) + year * (not (year == -1))
        month = datetime.datetime.today().month * (month == -1) + month * (not (month == -1))
        ix = self.transaction_data[(self.transaction_data.date.array.year == year) &
                                   (self.transaction_data.date.array.month == month)].index
        self.transaction_data.drop(labels=ix, inplace=inplace)

    @abstractmethod
    def update_from_raw_files(self):
        """Each account type has its own update method"""

    @abstractmethod
    def get_predicted_balance(self):
        """Each account type has its own prediction method"""

    def get_account_specific_codes(self, cashflow: pd.DataFrame, description_column="description") -> pd.Series:
        """
        This function returns a vector corresponding to all transaction code associated to the description vector given
        as argument

        args:
            - description   A vector of transaction description (often reffered to as "object, item, description, etc."
                            in bank statements). The relations between the descriptions and the codes are contained in
                            "assignations.csv"
        """
        descriptions = cashflow.loc[:, description_column]

        account_assignations_path = self.metadata.assignation_file_path
        if not os.path.exists(account_assignations_path):
            raise IOError(f"Assignation file not found at {account_assignations_path}")
        with open(account_assignations_path, 'r') as f:
            assignations = json.load(f)

        codes = list()
        for index, description in enumerate(descriptions):
            codes.append("na")
            code_headers = list()
            for code, assignations_list in assignations.items():
                code_headers.append(code)
                if assignations_list is not []:
                    for assignation in assignations_list:
                        if description.lower().find(assignation.lower()) != -1:
                            codes[len(codes) - 1] = code
                            break
        if codes[-1:] == ['na']:
            show_menu = True
            while show_menu:
                print_codes_menu(code_headers, cashflow.iloc[index].dropna().to_string())
                code = input()
                if code != 'na':
                    try:
                        code = int(code)
                        if code <= 0 or code > len(code_headers):
                            print(f'{colorama.Fore.RED}Error: {colorama.Fore.RESET}'
                                  f'Please enter a number between 1 and {len(code_headers)}')
                        else:
                            code = code_headers[code - 1]
                            codes[-1:] = [code]
                            show_menu = False
                    except ValueError:
                        print(f'{colorama.Fore.RED}Error: {colorama.Fore.RESET}'
                              f'Please enter a number between 1 and {len(assignations.columns)}')
                else:
                    show_menu = False

        return pd.Series(codes)


class DesjardinsOP(Account):
    def __init__(self, lmetadata: AccountMetadata):
        super().__init__(lmetadata=lmetadata)
        self.col_mask = ['date', 'transaction_num', 'description', 'withdrawal', 'deposit', 'balance', 'code']
        self.transaction_data = self.transaction_data[self.transaction_data['account'] == 'EOP']

    def get_summed_data(self, year=None, month=None, day=None):
        d = super().get_summed_data(year=year, month=month, day=day)
        if d.shape[0] > 0:
            d = d.loc[:, ['withdrawal', 'deposit']]
            d['total'] = np.subtract(d['deposit'], d['withdrawal'])
            if 'internal_cashflow' in d.index:
                d.drop(labels=['internal_cashflow'], inplace=True)
            d.drop(columns=['deposit', 'withdrawal'], inplace=True)

        return d

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['deposit'] - d['withdrawal']
            d.drop(columns=['withdrawal', 'deposit', 'balance', 'transaction_num'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):

        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if range_data.shape[0] > 0:
            withdrawals = range_data.loc[range_data.withdrawal > 0, :].copy()
            deposits = range_data.loc[range_data.deposit > 0, :].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'deposit') | (columns == 'code')
            withdrawals_ix = (columns == 'withdrawal') | (columns == 'code')
            withdrawals.drop(axis=1, columns=withdrawals.columns[~withdrawals_ix], inplace=True)
            deposits.drop(axis=1, columns=deposits.columns[~deposits_ix], inplace=True)

            withdrawals_counts = withdrawals.groupby(by='code').count()
            deposits_counts = deposits.groupby(by='code').count()

            withdrawals_freqs = np.divide(withdrawals_counts, delta)
            deposits_freqs = np.divide(deposits_counts, delta)

            daily_freqs = withdrawals_freqs.join(other=deposits_freqs, how='outer').replace(np.nan, 0)

            withdrawals_mean = withdrawals.groupby(by='code').mean()
            deposits_mean = deposits.groupby(by='code').mean()

            means = withdrawals_mean.join(other=deposits_mean, how='outer').replace(np.nan, 0)

            withdrawals_std = withdrawals.groupby(by='code').std()
            deposits_std = deposits.groupby(by='code').std()

            std_devs = withdrawals_std.join(other=deposits_std, how='outer').replace(np.nan, 0)

            daily_means = np.multiply(means, daily_freqs)
            daily_std = np.multiply(std_devs, daily_freqs)

            daily_means.withdrawal *= -1
            daily_freqs.withdrawal *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])
        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

    def update_from_raw_files(self):
        new_entries = 0
        csv_files = os.listdir(self.metadata.raw_files_dir)
        for x in csv_files:
            if x[-4:] == '.csv':
                new = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
                                  encoding='latin1',
                                  names=self.metadata.columns_names)
                for ix, row in new.iterrows():
                    # TODO transaction number-based duplication detection logic is not valid for
                    #  partial monthly statement downloads
                    if (row['date'], row['transaction_num']) in self.transaction_data.set_index(
                            keys=['date', 'transaction_num']).index:
                        continue
                    else:
                        if row['account'] == 'EOP':
                            row['date'] = pd.to_datetime(row['date'], format='%Y-%m-%d')
                            row['code'] = get_common_codes(pd.DataFrame(row).T).values[0]
                            if row['code'] == "na":
                                row['code'] = self.get_account_specific_codes(pd.DataFrame(row).T).values[0]
                            row.replace(np.nan, 0, inplace=True)
                            self.transaction_data = self.transaction_data.append(other=row, ignore_index=True)
                            new_entries += 1

        if new_entries > 0:
            self.transaction_data.drop_duplicates(keep='first', subset=self.metadata.columns_names, inplace=True)
            self.transaction_data.sort_values(by=['date', 'transaction_num'], inplace=True)

            print(f'{new_entries} new entries added')
            if self._save_prompt():
                self.to_pickle()


    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        df = self.get_data()
        if self.planned_transactions is not None:
            template = df.iloc[0, :].copy(deep=True)
            bal = self.get_current_balance()
            for d in range(0, days):
                date = datetime.datetime.today().date() + timedelta(days=d)
                for ix, ptransaction in self.planned_transactions.iterrows():
                    if _is_planned_transaction(year=ptransaction['year'],
                                               month=ptransaction['month'],
                                               day=ptransaction['day'],
                                               date=date):
                        template.loc['date'] = date
                        template['transaction_num'] = 'na'
                        template['description'] = ptransaction['description']
                        template['withdrawal'] = ptransaction['withdrawal']
                        template['deposit'] = ptransaction['deposit']
                        template['balance'] = bal + ptransaction['deposit'] - ptransaction['withdrawal']
                        template['code'] = 'planned'
                        df = df.append(other=template, ignore_index=True)
                        df.sort_values(by=['date', 'transaction_num'], inplace=True)

                        bal = template['balance']
        else:
            df = None
        return df

class DesjardinsMC(Account):
    def __init__(self, lmetadata: AccountMetadata):
        super().__init__(lmetadata=lmetadata)
        self.col_mask = ['date', 'transaction_num', 'description', 'interests', 'advance', 'reimboursment', 'balance',
                         'code']
        self.transaction_data = self.transaction_data[self.transaction_data['account'] == 'MC2']

    def get_summed_data(self, year=None, month=None, day=None):
        d = super().get_summed_data(year=year, month=month, day=day)
        if d.shape[0] > 0:
            if 'interest' in d.index:
                d.loc['interest', 'advance'] = d.loc['interest', 'interests']
            d = d.loc[:, ['advance', 'reimboursment']]
            d['total'] = np.subtract(d['reimboursment'], d['advance'])
            if 'internal_cashflow' in d.index:
                d.drop(labels=['internal_cashflow'], inplace=True)
            d.drop(columns=['advance', 'reimboursment'], inplace=True)

        return d

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['interests'].apply(lambda i: -i)
            d.drop(columns=['transaction_num', 'interests', 'advance', 'reimboursment', 'balance'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):

        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if range_data.shape[0] > 0:
            withdrawals = range_data.loc[(range_data.advance > 0) | (range_data.interests > 0), :].copy()
            deposits = range_data.loc[range_data.reimboursment > 0, :].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'reimboursment') | (columns == 'code')
            withdrawals_ix = (columns == 'interests') | (columns == 'advance') | (columns == 'code')
            withdrawals.drop(axis=1, columns=withdrawals.columns[~withdrawals_ix], inplace=True)
            deposits.drop(axis=1, columns=deposits.columns[~deposits_ix], inplace=True)

            withdrawals_counts = withdrawals.groupby(by='code').count()
            deposits_counts = deposits.groupby(by='code').count()

            withdrawals_freqs = np.divide(withdrawals_counts, delta)
            deposits_freqs = np.divide(deposits_counts, delta)

            daily_freqs = withdrawals_freqs.join(other=deposits_freqs, how='outer').replace(np.nan, 0)

            withdrawals_mean = withdrawals.groupby(by='code').mean()
            deposits_mean = deposits.groupby(by='code').mean()

            means = withdrawals_mean.join(other=deposits_mean, how='outer').replace(np.nan, 0)

            withdrawals_std = withdrawals.groupby(by='code').std()
            deposits_std = deposits.groupby(by='code').std()

            std_devs = withdrawals_std.join(other=deposits_std, how='outer').replace(np.nan, 0)

            daily_means = np.multiply(means, daily_freqs)
            daily_std = np.multiply(std_devs, daily_freqs)

            daily_means.interests *= -1
            daily_means.advance *= -1
            daily_freqs.interests *= -1
            daily_freqs.advance *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])
        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

    def get_current_balance(self) -> float:
        return self.transaction_data.tail(n=1)['balance'].values[0]

    def update_from_raw_files(self):
        new_entries = 0
        csv_files = os.listdir(self.metadata.raw_files_dir)
        for x in csv_files:
            if x[-4:] == '.csv':
                new = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
                                  encoding='latin1',
                                  names=self.metadata.columns_names)
                for ix, row in new.iterrows():
                    if (row['date'], row['transaction_num']) in self.transaction_data.set_index(
                            keys=['date', 'transaction_num']).index:
                        continue
                    else:
                        if row['account'] == 'MC2':
                            row['date'] = pd.to_datetime(row['date'], format='%Y-%m-%d')
                            row['code'] = get_common_codes(pd.DataFrame(row).T).values[0]
                            row.replace(np.nan, 0, inplace=True)
                            self.transaction_data = self.transaction_data.append(other=row, ignore_index=True)
                            new_entries += 1

        if new_entries > 0:
            self.transaction_data.drop_duplicates(keep='first', subset=self.metadata.columns_names, inplace=True)
            self.transaction_data.sort_values(by=['date', 'transaction_num'], inplace=True)

            print(f'{new_entries} new entries added')
            if self._save_prompt():
                self.to_pickle()

    def get_predicted_balance(self,
                              end_date: datetime.date,
                              sim_date: datetime.date = None,
                              force_new: bool = False,
                              avg_interval: int = 90,
                              montecarl_iterations=3) -> pd.DataFrame:
        """
        This function gives an estimation of the future balance of the account for a specified number of days. It
        uses the planned transaction data and optionally the average spending data.
        :param end_date: date at which the prediction ends
        :param sim_date: date at which the prediction starts
        :param avg_interval: The number of days over which the average and stdev are computed for probabilistic calcs.
        :return: a Dataframe containing the predicted balance of the account
        """

        fname = f'prd' + f'_{sim_date}' * (sim_date is not None) + f'_{end_date}.pkl'
        fpath = os.path.join(pickle_dir, fname)
        if os.path.exists(fpath) and not force_new:
            print(f'Prediction loaded from previous execution ({fpath})')
            with open(fpath, 'rb') as file:
                df = pickle.load(file=file)
        else:
            # if no sim_date is specified, the prediction starts from the last entry of transaction data
            df = self.get_data().copy()
            if sim_date is None:
                idate = pd.to_datetime(df.tail(1)['date'].values).date[0]
            else:
                if sim_date > end_date:
                    raise RuntimeError(f'Simulation starting date ({sim_date}) is ulterior to simulation ending date '
                                       f'({end_date})')
                idate = sim_date
                df.drop(labels=df[df['date'] > str(sim_date)].index, inplace=True)

            # Get the average, std dev. and daily freqencies of transaction for the specified time interval
            sdate = idate - datetime.timedelta(days=avg_interval)
            avg, std, freqs = accounts.get_data_range_daily_average(start_date=sdate, end_date=idate)

            # balances_list = list()
            balances_list = [df.tail(1).balance.values[0]].copy()
            initial_bal_value = balances_list[0]
            transaction_tuples = {'date': list(),
                                  'transaction_num': list(),
                                  'description': list(),
                                  'interests': list(),
                                  'advance': list(),
                                  'reimboursment': list(),
                                  'balance': list(),
                                  'code': list()}

            # Monte carlo iterations
            t_num_mc_prefix = 0
            digit_num = int(np.log10(montecarl_iterations)) + 1
            for mc_it in tqdm(range(0, montecarl_iterations),
                          position=0,
                          leave=False,
                          desc='Monte-Carlo iterations'):
                # for each day in the projection
                for delta_days in tqdm(range(1, (end_date-idate).days),
                                       position=1,
                                       leave=False,
                                       desc='Projected days'):
                    date = (idate + timedelta(days=delta_days))
                    t_num_date_prefix = 0

                    # add planned transaction tuple
                    for ix, ptransaction in self.planned_transactions.iterrows():
                        if _is_planned_transaction(year=ptransaction['year'],
                                                   month=ptransaction['month'],
                                                   day=ptransaction['day'],
                                                   date=date):

                            transaction_tuples['date'].append(pd.to_datetime(date))
                            transaction_tuples['transaction_num'].append(f'{t_num_mc_prefix:0{digit_num}d}_'
                                                                         f'{t_num_date_prefix:03d}_'
                                                                         f'planned')
                            transaction_tuples['description'].append(ptransaction['description'])
                            transaction_tuples['interests'].append(0)
                            transaction_tuples['advance'].append(ptransaction['withdrawal'])
                            transaction_tuples['reimboursment'].append(ptransaction['deposit'])
                            balance = balances_list[-1:][0] + ptransaction['withdrawal'] - ptransaction['deposit']
                            transaction_tuples['balance'].append(balance)
                            transaction_tuples['code'].append(ptransaction['code'])

                            balances_list.append(balance)
                            t_num_date_prefix += 1

                    # add average expenses to prediction
                    for expense_code in avg.index:
                        if expense_code.lower() not in ['internet', 'rent', 'pay',
                                                        'cell', 'hydro', 'interest', 'other',
                                                        'credit', 'impots & tps', 'trading']:

                            # if mean_inscribed:
                            # compute probabilistic threshold of transaction occurence
                            amount = random.gauss(mu=avg.loc[expense_code, 'total_mean'] / freqs.loc[expense_code, 'total_freq'],
                                                  sigma=std.loc[expense_code, 'total_std'] / freqs.loc[expense_code, 'total_freq'])
                            # amount = avg.loc[expense_code, 'total_mean'] / freqs.loc[expense_code, 'total_freq']
                            amount *= np.sign(avg.loc[expense_code, 'total_mean'])
                            amount = round(amount, 2)
                            thresh_prob = random.uniform(0, 1)
                            event_prob = abs(freqs.loc[expense_code, 'total_freq'])
                            amount = amount * (thresh_prob < event_prob)
                            amount = min(amount, 0) * (avg.loc[expense_code, 'total_mean'] < 0) + \
                                     max(amount, 0) * (avg.loc[expense_code, 'total_mean'] >= 0)

                            if amount != 0:
                                transaction_tuples['date'].append(pd.to_datetime(date))
                                transaction_tuples['transaction_num'].append(f'{t_num_mc_prefix:0{digit_num}d}_'
                                                                             f'{t_num_date_prefix:03d}_'
                                                                             f'mc_expected')
                                transaction_tuples['description'].append(expense_code)
                                transaction_tuples['interests'].append(0)
                                transaction_tuples['advance'].append(
                                    abs(amount) * (avg.loc[expense_code, 'total_mean'] < 0))
                                transaction_tuples['reimboursment'].append(
                                    abs(amount) * (avg.loc[expense_code, 'total_mean'] >= 0))
                                balance = balances_list[-1:][0] - transaction_tuples['reimboursment'][-1:][0] + \
                                          transaction_tuples['advance'][-1:][0]
                                transaction_tuples['balance'].append(balance)
                                transaction_tuples['code'].append(expense_code)

                                balances_list.append(balance)
                                t_num_date_prefix += 1

                            if mc_it == 0:
                                amount = round(avg.loc[expense_code, 'total_mean'], 2)

                                transaction_tuples['date'].append(pd.to_datetime(date))
                                transaction_tuples['transaction_num'].append(f'{t_num_mc_prefix:0{digit_num}d}_'
                                                                             f'{t_num_date_prefix:03d}_'
                                                                             f'mean_expected')
                                transaction_tuples['description'].append(expense_code)
                                transaction_tuples['interests'].append(0)
                                transaction_tuples['advance'].append(abs(amount) * (avg.loc[expense_code, 'total_mean'] < 0))
                                transaction_tuples['reimboursment'].append(abs(amount) * (avg.loc[expense_code, 'total_mean'] >= 0))
                                balance = balances_list[-1:][0] - transaction_tuples['reimboursment'][-1:][0] + transaction_tuples['advance'][-1:][0]
                                transaction_tuples['balance'].append(balance)
                                transaction_tuples['code'].append(expense_code)

                                balances_list.append(balance)
                                t_num_date_prefix += 1

                if montecarl_iterations > 1:
                    # insert montecarlo reset tuple
                    transaction_tuples['date'].append(pd.to_datetime(idate))
                    transaction_tuples['transaction_num'].append(f'montecarlo_reset')
                    transaction_tuples['description'].append(expense_code)
                    transaction_tuples['interests'].append(0)
                    transaction_tuples['advance'].append(1)
                    transaction_tuples['reimboursment'].append(1)
                    balance = initial_bal_value
                    transaction_tuples['balance'].append(balance)
                    transaction_tuples['code'].append('na')

                    balances_list.append(initial_bal_value)
                    t_num_mc_prefix += 1

            df_pred = pd.DataFrame(transaction_tuples)

            # add interest estimate
            df_pred.sort_values(by=['date', 'transaction_num'], inplace=True)
            df_pred['delta'] = df_pred.date.diff().shift(-1)
            df_pred['delta'] = df_pred.delta.array.days
            df_pred['cap_interest'] = np.multiply(df_pred.balance, df_pred.delta) * self.metadata.interest_rate / 365
            df_pred.loc[df_pred['cap_interest'] < 0, 'cap_interest'] = 0
            df_pred.replace(np.nan, 0, inplace=True)
            df_pred.reset_index(drop=True, inplace=True)

            interest_sum = 0
            itt = iter(df_pred.iterrows())
            for ix, row in itt:
                interest_sum += row['cap_interest']
                if row['date'].day == 1 and row['interests'] == 0 and row['code'] == 'interest':
                    while row['date'].day == 1:
                        if row['code'] == 'interest' and row['interests'] == 0:
                            df_pred.loc[ix, 'interests'] = interest_sum
                            df_pred.loc[ix, 'balance'] += interest_sum
                        ix, row = next(itt)
                    interest_sum = 0

            df = pd.concat([df, df_pred]).sort_values(by=['date', 'transaction_num'])
            df.dropna(axis=0, how='all', inplace=True)
            ix = (df['interests'] == 0) &\
                 (df['advance'] == 0) &\
                 (df['reimboursment'] == 0) &\
                 (df['transaction_num'] == "expected")
            df.drop(labels=df.index[ix], axis=0, inplace=True)
            df.sort_values(by=['date', 'transaction_num'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            with open(os.path.join(pickle_dir, fname), 'wb') as file:
                pickle.dump(df, file=file)

        return df

    def plot(self, year=None, month=None, day=None, show=False, figsize=(7, 8)):
        d = self.get_data_by_date(year=year, month=month, day=day).copy()
        d.loc[:, 'date'] = d.loc[:, 'date'].apply(lambda i: str(i).replace(' 00:00:00', ''))
        d.loc[:, 'balance'] = -1 * d.loc[:, 'balance'].copy()

        fig = plt.figure(figsize=figsize)
        sns.pointplot(x='date', y='balance', data=d, color='green')
        plt.title(f'{year}-{month}')

        plt.xticks(rotation=90)
        if show:
            plt.show()
        return fig

    def plot_prediction_compare(self,
                                end_date: datetime.date,
                                start_date: datetime.date = None,
                                sim_dates: List[Union[datetime.date, None]] = None,
                                show=False,
                                fig_size=(7, 8),
                                force_new: bool = False,
                                avg_interval: int = 90,
                                montecarl_iterations=100) -> None:
        """

        :param end_date:        The simulation end date.
        :param start_date:      The simulation start date.
        :param sim_dates:      A list of dates from which a simulation starts even though actual data follows that date
        :param show:            Show the plot, as opposed to returning only the canvas.
        :param fig_size:        The size of the figure in inches.
        :param force_new:       Force a new simulation even though a preceding simulation has been run wih the
                                same dates.
        :param avg_interval:    The interval of days preceeding the simulation start over which the mean and std dev
                                are computed for each transaction code.
        """

        def plot_predictions(predictions: List[pd.DataFrame]):
            for ix, pred in enumerate(tqdm(predictions, desc='ploting predictions...')):
                fill_colors = (random.random(), random.random(), random.random())
                passed_ix = ~(pred["transaction_num"].str.contains("planned") |
                              (pred["transaction_num"].str.contains("expected")) |
                              (pred["transaction_num"].str.contains("montecarlo")))

                mc_futur_ix = ~passed_ix & (pred["transaction_num"].str.contains("mc") | pred["transaction_num"].str.contains("montecarlo"))
                mean_futur_ix = ~passed_ix & (pred["transaction_num"].str.contains("mean") | pred["transaction_num"].str.contains("montecarlo"))

                df_mean = pred[np.where(mc_futur_ix == True)[0][0] - 1:].copy().groupby(by='date').mean(numeric_only=True)
                df_std = pred[np.where(mc_futur_ix == True)[0][0] - 1:].copy().groupby(by='date').std(numeric_only=True)
                plt.fill_between(x=pd.to_datetime(df_std.index),
                                 y1=df_mean.balance - 3 * df_std.balance,
                                 y2=df_mean.balance + 3 * df_std.balance,
                                 color=fill_colors,
                                 alpha=0.45)
                plt.plot(pd.to_datetime(df_std.index),
                         df_mean.balance,
                         color=fill_colors,
                         linestyle='--')

                # we want to plot past transaction only once
                if ix == len(predictions) - 1:
                    # sns.lineplot(data=pred[passed_ix].copy(), x='date', y='balance', color='g')
                    plt.plot(pd.to_datetime(pred[passed_ix].date), pred[passed_ix].balance, color='g')

        predictions: List[pd.DataFrame] = list()

        if sim_dates is None:
            sim_dates = [None]
        else:
            sim_dates.append(None)

        for sdate in sim_dates:
            prediction = self.get_predicted_balance(end_date=end_date,
                                              sim_date=sdate,
                                              force_new=force_new,
                                              avg_interval=avg_interval,
                                              montecarl_iterations=montecarl_iterations).copy()
            prediction.loc[:, 'balance'] = -1 * prediction.loc[:, 'balance']
            if start_date is None:
                start_date = datetime.datetime.today().date() - timedelta(days=7)
            prediction = prediction[prediction['date'].array.date > start_date]
            prediction.date = prediction.date.apply(func=lambda i: str(i).replace(' 00:00:00', ''))
            predictions.append(prediction)

        title = f'Prediction {start_date} to {end_date}'
        fig = plt.figure(figsize=fig_size, num=title)

        plot_predictions(predictions=predictions)

        plt.title(title)
        plt.grid(b=True, which='major', axis='both')
        plt.grid(b=True, which='minor', axis='x')
        plt.xticks(rotation=90)
        min_y = min([estim.balance.min() for estim in predictions])
        max_y = max([estim.balance.max() for estim in predictions])
        plt.yticks(ticks=np.arange(np.round(min_y - 1000, -3),
                                   np.round(max_y + 1000, -3),
                                   500))

        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        if show:
            plt.show()
        return fig

    def _get_last_interest_payment_date(self) -> datetime.date:
        return (self.transaction_data[self.transaction_data['interests'] > 0]).tail(n=1).date.array.date[0]

    def _get_last_interest_payment_index(self, sim_date: datetime.date = None) -> datetime.date:
        if sim_date is not None:
            res = (self.transaction_data[(self.transaction_data['interests'] > 0) &
                                         (self.transaction_data['date'].array.date < sim_date)]).tail(n=1).index[0]
        else:
            res = (self.transaction_data[self.transaction_data['interests'] > 0]).tail(n=1).index[0]
        return res

    def _get_next_interest_payment_date(self):
        d = self._get_last_interest_payment_date()
        d2 = d + relativedelta(months=+1)
        # no transactions on saturday or sunday, instead it is reported to next monday
        while d2.isoweekday() in [6, 7]:
            d2 = d2 + timedelta(days=1)
        return (self.transaction_data[self.transaction_data['interests'] > 0]).tail(n=1).date.array.date[0]


class DesjardinsPR(Account):
    def __init__(self, lmetadata: AccountMetadata):
        super().__init__(lmetadata=lmetadata)
        self.col_mask = ['date', 'transaction_num', 'description', 'capital_paid', 'reimboursment', 'balance', 'code']
        self.transaction_data = self.transaction_data[self.transaction_data['account'] == 'PR1']

    def get_summed_data(self, year=None, month=None, day=None):
        d = self.get_data(year=year, month=month, day=day)
        if d.shape[0] > 0:
            d = d.loc[:, ['capital_paid', 'reimboursment', 'code']]
            d = d.groupby('code').sum()
            d['total'] = np.subtract(d.loc[:, ['capital_paid']], np.asarray(d.loc[:, ['reimboursment']]))
            if 'internal_cashflow' in d.index:
                d.drop(labels='internal_cashflow', inplace=True)
            d.drop(columns=['capital_paid', 'reimboursment'], inplace=True)

        return d

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['capital_paid'] - d['reimboursment']
            d.drop(columns=['transaction_num', 'capital_paid', 'reimboursment', 'balance'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if range_data.shape[0] > 0:

            withdrawals = range_data.loc[range_data.advance > 0, :].copy()
            deposits = range_data.loc[range_data.reimboursment > 0, :].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'reimboursment') | (columns == 'code')
            withdrawals_ix = (columns == 'advance') | (columns == 'code')
            withdrawals.drop(axis=1, columns=withdrawals.columns[~withdrawals_ix], inplace=True)
            deposits.drop(axis=1, columns=deposits.columns[~deposits_ix], inplace=True)

            withdrawals_counts = withdrawals.groupby(by='code').count()
            deposits_counts = deposits.groupby(by='code').count()

            withdrawals_freqs = np.divide(withdrawals_counts, delta)
            deposits_freqs = np.divide(deposits_counts, delta)

            daily_freqs = withdrawals_freqs.join(other=deposits_freqs, how='outer').replace(np.nan, 0)

            withdrawals_mean = withdrawals.groupby(by='code').mean()
            deposits_mean = deposits.groupby(by='code').mean()

            means = withdrawals_mean.join(other=deposits_mean, how='outer').replace(np.nan, 0)

            withdrawals_std = withdrawals.groupby(by='code').std()
            deposits_std = deposits.groupby(by='code').std()

            std_devs = withdrawals_std.join(other=deposits_std, how='outer').replace(np.nan, 0)

            daily_means = np.multiply(means, daily_freqs)
            daily_std = np.multiply(std_devs, daily_freqs)

            daily_means.advance *= -1
            daily_freqs.advance *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])
        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

    def get_current_balance(self) -> float:
        return self.transaction_data.tail(n=1)['balance'].values[0]

    def update_from_raw_files(self):
        new_entries = 0
        csv_files = os.listdir(self.metadata.raw_files_dir)
        for x in csv_files:
            if x[-4:] == '.csv':
                new = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
                                  encoding='latin1',
                                  names=self.metadata.columns_names)
                for ix, row in new.iterrows():
                    if (row['date'], row['transaction_num']) in self.transaction_data.set_index(
                            keys=['date', 'transaction_num']).index:
                        continue
                    else:
                        if row['account'] == 'PR1':
                            row['date'] = pd.to_datetime(row['date'], format='%Y-%m-%d')
                            row['code'] = get_common_codes(pd.DataFrame(row).T).values[0]
                            row.replace(np.nan, 0, inplace=True)
                            self.transaction_data = self.transaction_data.append(other=row, ignore_index=True)
                            new_entries += 1

        if new_entries > 0:
            self.transaction_data.drop_duplicates(keep='first', subset=self.metadata.columns_names, inplace=True)
            self.transaction_data.sort_values(by=['date', 'transaction_num'], inplace=True)

            print(f'{new_entries} new entries added')
            if self._save_prompt():
                self.to_pickle()

    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        df = self.get_data()
        if self.planned_transactions is not None:
            template = df.iloc[0, :].copy(deep=True)
            bal = self.get_current_balance()
            for d in range(0, days):
                date = (datetime.datetime.today().date() + timedelta(days=d))
                for ix, ptransaction in self.planned_transactions.iterrows():
                    if _is_planned_transaction(year=ptransaction['year'],
                                               month=ptransaction['month'],
                                               day=ptransaction['day'],
                                               date=date):
                        template.loc['date'] = date
                        template['transaction_num'] = 'na'
                        template['description'] = ptransaction['description']
                        template['paid_capital'] = ptransaction['withdrawal']
                        template['reimboursment'] = ptransaction['deposit']
                        template['balance'] = bal + ptransaction['paid_capital'] - ptransaction['reimboursment']
                        template['code'] = 'planned'
                        df = df.append(other=template, ignore_index=True)
                        df.sort_values(by=['date', 'transaction_num'], inplace=True)

                        bal = template['balance']
        else:
            df = None
        return df


class VisaPP(Account):
    def __init__(self, lmetadata: AccountMetadata):
        super().__init__(lmetadata=lmetadata)
        self.col_mask = ['date', 'transaction_num', 'description', 'credit/payment', 'balance', 'code']

    def _load_from_raw_pdf_files(self) -> pd.DataFrame:
        # pdf_files = Path(input_path).glob('*.pdf')
        pdf_files = os.listdir(self.metadata.raw_files_dir)

        # Desjardins ppcard pdf files columns names
        lnames = ['date', 'transaction_num', 'description', 'credit/payment']
        cash_flow = pd.DataFrame(columns=lnames)

        suffix1 = '_processed.txt'
        for x in pdf_files:
            if x[-len(suffix1):] == suffix1:
                tot_df = pd.read_csv(os.path.join(self.metadata.raw_files_dir, x),
                                     sep=';',
                                     encoding='utf-8',
                                     names=lnames,
                                     header=None)
                start_index = tot_df[tot_df['date'] == 'Jour'].index[0] + 1
                mid_index = tot_df[tot_df['date'] == 'Total'].index[0]
                end_index = tot_df[tot_df['date'] == 'SOLDE PRÉCÉDENT'].index[0]
                # year = int(tot_df.iloc[1, 1])

                expenses = tot_df[start_index:mid_index].copy()
                payments = tot_df[mid_index + 1:end_index].copy()

                expenses.loc[:, 'transaction_num'] = expenses.loc[:, 'transaction_num'].apply(
                    lambda li: str(int(li)) + 'e')
                payments.loc[:, 'transaction_num'] = payments.loc[:, 'transaction_num'].apply(
                    lambda li: str(int(li)) + 'p')

                cash_flow = cash_flow.append(expenses, ignore_index=True)
                cash_flow = cash_flow.append(payments, ignore_index=True)

            elif x[-4:] == '.txt':
                tnumber_prefixes, ix = ['e', 'p'], 0
                with open(os.path.join(self.metadata.raw_files_dir, x), 'r') as raw_file:
                    for i, line in enumerate(raw_file):
                        print(line)
                        if line.find('Date de transaction') == 0:
                            for t in raw_file:
                                if t.find('Total :') == 0 or t.find('nullDétail') == 0 or t.find('\n') == 0:
                                    break
                                res = t.split(sep='\t')
                                res[0] = res[0].replace('JUI', 'JUL')
                                res[0] = res[0].replace('MAI', 'MAY')
                                date = pd.to_datetime(res[0])
                                t_number = res[4].replace('0', '')
                                t_number = t_number + tnumber_prefixes[ix]
                                description = res[7]
                                transaction = res[8 + ix]
                                r1 = re.compile(pattern='^([0-9]+),([0-9]{1,2})')
                                r2 = re.compile(pattern='^CR ([0-9]+),([0-9]{1,2})')
                                transaction = r1.sub(repl=r'-\1.\2', string=transaction)
                                transaction = r2.sub(repl=r'\1.\2', string=transaction)
                                cash_flow = cash_flow.append(other=pd.Series([date,
                                                                              t_number,
                                                                              description,
                                                                              transaction],
                                                                             index=lnames),
                                                             ignore_index=True)
                            ix += 1
            elif x[-4:] == '.csv':
                df = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
                                 encoding='latin1',
                                 names=self.metadata.columns_names)
                df.dropna(axis=1, how='all', inplace=True)
                df.loc[df['description'] == 'PAIEMENT CAISSE', 'transaction_num'] = \
                    df.loc[df['description'] == 'PAIEMENT CAISSE', 'transaction_num'].apply(lambda i: str(int(i)) + 'p')
                df.loc[~(df['description'] == 'PAIEMENT CAISSE'), 'transaction_num'] =\
                    df.loc[~(df['description'] == 'PAIEMENT CAISSE'), 'transaction_num'].apply(lambda i: str(int(i)) + 'e')
                df.replace(np.nan, 0, inplace=True)
                df['payment/credit'] = df['payment/credit'] - df['credit']
                df.drop(columns=['acc_name', 'credit'], inplace=True)

        # Adds column,and inputs transaction code
        cash_flow = cash_flow.replace(np.nan, 0)
        cash_flow['date'] = pd.to_datetime(cash_flow['date'])
        cash_flow.sort_values(by=['date', 'transaction_num'], ascending=True, inplace=True)
        cash_flow['credit/payment'] = pd.to_numeric(cash_flow['credit/payment'])
        cash_flow['balance'] = cash_flow['credit/payment'].cumsum()
        cash_flow.reset_index(inplace=True, drop=True)
        cash_flow['code'] = get_common_codes(cash_flow)

        return cash_flow

    def get_summed_data(self, year=None, month=None, day=None):
        d = self.get_data(year=year, month=month, day=day)
        if d.shape[0] > 0:
            d = d.loc[:, ['credit/payment', 'code']].groupby('code').sum()
            if 'internal_cashflow' in d.index:
                d.drop(labels='internal_cashflow', inplace=True)
            d.columns = ['total']
        return d

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['credit/payment']
            d.drop(columns=['credit/payment', 'balance'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if range_data.shape[0] > 0:
            withdrawals = range_data[range_data['credit/payment'] < 0].copy()
            deposits = range_data[range_data['credit/payment'] > 0].copy()

            columns = withdrawals.columns
            ix = (columns == 'credit/payment') | (columns == 'code')
            withdrawals.drop(axis=1, columns=withdrawals.columns[~ix], inplace=True)
            deposits.drop(axis=1, columns=deposits.columns[~ix], inplace=True)

            withdrawals_counts = withdrawals.groupby(by='code').count()
            deposits_counts = deposits.groupby(by='code').count()

            withdrawals_freqs = np.divide(withdrawals_counts, delta)
            deposits_freqs = np.divide(deposits_counts, delta)

            daily_freqs = withdrawals_freqs.join(other=deposits_freqs, how='outer', lsuffix="_out").replace(np.nan, 0)

            withdrawals_mean = withdrawals.groupby(by='code').mean()
            deposits_mean = deposits.groupby(by='code').mean()

            means = withdrawals_mean.join(other=deposits_mean, how='outer', lsuffix="_out").replace(np.nan, 0)

            withdrawals_std = withdrawals.groupby(by='code').std()
            deposits_std = deposits.groupby(by='code').std()

            std_devs = withdrawals_std.join(other=deposits_std, how='outer', lsuffix='_out').replace(np.nan, 0)

            daily_means = np.multiply(means, daily_freqs)
            daily_std = np.multiply(std_devs, daily_freqs)

            daily_freqs['credit/payment_out'] *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])

        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

    def get_current_balance(self) -> float:
        return self.transaction_data.tail(n=1)['balance'].values[0]

    def update_from_raw_files(self):
        raise RuntimeError(f"This account is: {self.metadata.status.name}")

    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        df = self.get_data()
        return df


class CapitalOne(Account):
    def __init__(self, lmetadata: AccountMetadata):
        super().__init__(lmetadata=lmetadata)
        self.col_mask = ['date', 'description', 'debit', 'credit', 'code']

    def _load_from_raw_files(self, update_data=False) -> None:
        super()._load_from_raw_files(update_data=update_data)

    def get_summed_data(self, year=None, month=None, day=None):
        d = self.get_data(year=year, month=month, day=day)
        if d.shape[0] > 0:
            d = d.loc[:, ['debit', 'credit', 'code']].groupby('code').sum()
            if 'internal_cashflow' in d.index:
                d.drop(labels='internal_cashflow', inplace=True)
            d.dropna(inplace=True)
            d['total'] = np.subtract(d.loc[:, 'credit'], d.loc[:, 'debit'])
            d.drop(columns=['debit', 'credit'], inplace=True)
        return d

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['credit'] - d['debit']
            d.drop(columns=['credit', 'debit'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if range_data.shape[0] > 0:
            withdrawals = range_data[range_data.debit > 0].copy()
            deposits = range_data[range_data.credit > 0].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'credit') | (columns == 'code')
            withdrawals_ix = (columns == 'debit') | (columns == 'code')
            withdrawals.drop(axis=1, columns=withdrawals.columns[~withdrawals_ix], inplace=True)
            deposits.drop(axis=1, columns=deposits.columns[~deposits_ix], inplace=True)

            withdrawals_counts = withdrawals.groupby(by='code').count()
            deposits_counts = deposits.groupby(by='code').count()

            withdrawals_freqs = np.divide(withdrawals_counts, delta)
            deposits_freqs = np.divide(deposits_counts, delta)

            daily_freqs = withdrawals_freqs.join(other=deposits_freqs, how='outer').replace(np.nan, 0)

            withdrawals_mean = withdrawals.groupby(by='code').mean()
            deposits_mean = deposits.groupby(by='code').mean()

            means = withdrawals_mean.join(other=deposits_mean, how='outer').replace(np.nan, 0)

            withdrawals_std = withdrawals.groupby(by='code').std()
            deposits_std = deposits.groupby(by='code').std()

            std_devs = withdrawals_std.join(other=deposits_std, how='outer').replace(np.nan, 0)

            daily_means = np.multiply(means, daily_freqs)
            daily_std = np.multiply(std_devs, daily_freqs)

            daily_means.debit *= -1
            daily_freqs.debit *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])

        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

    def update_from_raw_files(self):
        cash_flow = self.transaction_data
        old_entries_num = cash_flow.shape[0]
        if cash_flow is None:
            cash_flow = pd.DataFrame(columns=self.metadata.columns_names)

        csv_files = os.listdir(self.metadata.raw_files_dir)
        for x in csv_files:
            if x[-4:] == '.csv':
                tmp_df = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
                                     encoding='latin1',
                                     names=self.metadata.columns_names)
                tmp_df.drop(labels=[0], inplace=True)
                tmp_df.drop(columns=['posted_date'], inplace=True)
                cash_flow = cash_flow.append(tmp_df, ignore_index=True)

        # Convert strings to actual date time objects
        if 'date' in cash_flow.columns:
            cash_flow['date'] = pd.to_datetime(cash_flow['date'], format='%Y-%m-%d')

        # Adds column,and inputs transaction code
        cash_flow = cash_flow.replace(np.nan, 0)
        cash_flow.sort_values(by=['date'], inplace=True)
        cash_flow.reset_index(inplace=True, drop=True)
        cash_flow['credit'] = cash_flow['credit'].apply(lambda i: float(i))
        cash_flow['debit'] = cash_flow['debit'].apply(lambda i: float(i))
        cash_flow.drop_duplicates(subset=cash_flow.columns[cash_flow.columns != 'code'], inplace=True)
        for ix, row in cash_flow[cash_flow['code'] == 0].iterrows():
            cash_flow.loc[ix, 'code'] = get_common_codes(pd.DataFrame(row).T).values[0]

        new_entries = cash_flow.shape[0] - old_entries_num
        if new_entries > 0:
            self.transaction_data = cash_flow
            print(f'{new_entries} new entries added')
            if self._save_prompt():
                self.to_pickle()

    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        raise NotImplementedError("get_predicted_balance is not implemented for Capital One")


class CIBC(Account):
    def __init__(self, lmetadata: AccountMetadata):
        super().__init__(lmetadata=lmetadata)
        self.col_mask = ['date', 'description', 'debit', 'credit', 'code']

    def _load_from_raw_files(self, update_data=False) -> None:
        super()._load_from_raw_files(update_data=update_data)

    def get_summed_data(self, year=None, month=None, day=None):
        d = self.get_data(year=year, month=month, day=day)
        if d.shape[0] > 0:
            d = d.loc[:, ['debit', 'credit', 'code']].groupby('code').sum()
            if 'internal_cashflow' in d.index:
                d.drop(labels='internal_cashflow', inplace=True)
            d.dropna(inplace=True)
            d['total'] = np.subtract(d.loc[:, 'credit'], d.loc[:, 'debit'])
            d.drop(columns=['debit', 'credit'], inplace=True)
        return d

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['credit'] - d['debit']
            d.drop(columns=['credit', 'debit'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if range_data.shape[0] > 0:
            withdrawals = range_data.loc[range_data.debit > 0, :].copy()
            deposits = range_data.loc[range_data.credit > 0, :].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'credit') | (columns == 'code')
            withdrawals_ix = (columns == 'debit') | (columns == 'code')
            withdrawals.drop(axis=1, columns=withdrawals.columns[~withdrawals_ix], inplace=True)
            deposits.drop(axis=1, columns=deposits.columns[~deposits_ix], inplace=True)

            withdrawals_counts = withdrawals.groupby(by='code').count()
            deposits_counts = deposits.groupby(by='code').count()

            withdrawals_freqs = np.divide(withdrawals_counts, delta)
            deposits_freqs = np.divide(deposits_counts, delta)

            daily_freqs = withdrawals_freqs.join(other=deposits_freqs, how='outer').replace(np.nan, 0)

            withdrawals_mean = withdrawals.groupby(by='code').mean()
            deposits_mean = deposits.groupby(by='code').mean()

            means = withdrawals_mean.join(other=deposits_mean, how='outer').replace(np.nan, 0)

            withdrawals_std = withdrawals.groupby(by='code').std()
            deposits_std = deposits.groupby(by='code').std()

            std_devs = withdrawals_std.join(other=deposits_std, how='outer').replace(np.nan, 0)

            daily_means = np.multiply(means, daily_freqs)
            daily_std = np.multiply(std_devs, daily_freqs)

            daily_means.debit *= -1
            daily_freqs.debit *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])
        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

    def update_from_raw_files(self):
        cash_flow = self.transaction_data
        old_entries_num = cash_flow.shape[0]
        if cash_flow is None:
            cash_flow = pd.DataFrame(columns=self.metadata.columns_names)

        csv_files = os.listdir(self.metadata.raw_files_dir)
        for x in csv_files:
            if x[-4:] == '.csv':
                tmp_df = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
                                     encoding='latin1',
                                     names=self.metadata.columns_names)
                cash_flow = cash_flow.append(tmp_df, ignore_index=True)

        # Convert strings to actual date time objects
        if 'date' in cash_flow.columns:
            cash_flow['date'] = pd.to_datetime(cash_flow['date'], format='%Y-%m-%d')

        # Adds column,and inputs transaction code
        cash_flow = cash_flow.replace(np.nan, 0)
        cash_flow.sort_values(by=['date'], inplace=True)
        cash_flow.reset_index(inplace=True, drop=True)
        cash_flow['credit'] = cash_flow['credit'].apply(lambda i: float(i))
        cash_flow['debit'] = cash_flow['debit'].apply(lambda i: float(i))
        cash_flow.drop_duplicates(subset=cash_flow.columns[cash_flow.columns != 'code'], inplace=True)
        for ix, row in cash_flow[cash_flow['code'] == 0].iterrows():
            cash_flow.loc[ix, 'code'] = get_common_codes(pd.DataFrame(row).T).values[0]
            if cash_flow.loc[ix, 'code'] == "na":
                cash_flow.loc[ix, 'code'] = self.get_account_specific_codes(pd.DataFrame(row).T).values[0]

        new_entries = cash_flow.shape[0] - old_entries_num
        if new_entries > 0:
            self.transaction_data = cash_flow
            print(f'{new_entries} new entries added')
            if self._save_prompt():
                self.to_pickle()

    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        pass


class PayPal(Account):
    def __init__(self, lmetadata: AccountMetadata):
        super().__init__(lmetadata=lmetadata)
        self.col_mask = ['date', 'description', 'currency', 'amount', 'balance', 'code']

    # TTODO refactor to load data in parent class and process it
    # def _load_from_raw_files(self) -> pd.DataFrame:
    #     if self.metadata.raw_files_dir == '':
    #         raise ValueError('please specify a path for Desjardins CSV files')
    #
    #     csv_files = os.listdir(self.metadata.raw_files_dir)
    #     cash_flow = pd.DataFrame()
    #     for x in csv_files:
    #         if x[-4:].lower() == '.csv':
    #             cash_flow = cash_flow.append(
    #                 pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
    #                             encoding='latin1'),
    #                 ignore_index=True)
    #
    #     # Convert strings to actual date time objects
    #     cash_flow.columns = ['date', *cash_flow.columns[1:].str.lower()]
    #     if 'date' in cash_flow.columns:
    #         cash_flow['date'] = pd.to_datetime(cash_flow['date'], format='%d/%m/%Y')
    #     cash_flow.sort_values(by=['date'], inplace=True)
    #
    #     # Adds column,and inputs transaction code
    #     cash_flow.insert(loc=5,
    #                      column='description',
    #                      value=cash_flow['name'].replace(np.nan, 'NA') + ' - ' + cash_flow['type'])
    #     cash_flow.drop(columns=['name', 'type'], inplace=True)
    #     cash_flow = cash_flow.replace(np.nan, 0)
    #
    #     for ix, row in cash_flow.iterrows():
    #         if row['status'] == 'Completed' and row['currency'] != 'CAD' and row['balance'] != 0:
    #             if cash_flow.loc[ix + 1, 'status'] == 'Pending':
    #                 cash_flow.loc[ix, 'amount'] = -cash_flow.loc[ix + 1, 'amount']
    #                 cash_flow.loc[ix, 'balance'] = -cash_flow.loc[ix + 1, 'balance']
    #             else:
    #                 cash_flow.loc[ix, 'amount'] = np.sign(cash_flow.loc[ix, 'amount']) * abs(
    #                     cash_flow.loc[ix + 1, 'amount'])
    #                 cash_flow.loc[ix, 'balance'] = np.sign(cash_flow.loc[ix, 'amount']) * abs(
    #                     cash_flow.loc[ix + 1, 'balance'])
    #             cash_flow.loc[ix, 'currency'] = 'CAD'
    #         elif row['status'] == 'Completed' \
    #                 and row['description'][-len('Express Checkout Payment'):] == 'Express Checkout Payment' \
    #                 and row['currency'] == 'CAD' \
    #                 and row['balance'] == 0:
    #             cash_flow.loc[ix, 'balance'] = cash_flow.loc[ix, 'amount']
    #         elif row['status'] == 'Completed' \
    #                 and (row['description'] == 'NA - General Credit Card Deposit' or
    #                      row['description'] == 'NA - Reversal of ACH Deposit') \
    #                 and row['balance'] != 0:
    #             cash_flow.loc[ix, 'balance'] = 0
    #
    #     cash_flow.drop_duplicates(inplace=True)
    #     cash_flow.drop(index=cash_flow[~((cash_flow['balance'] != 0) & (cash_flow['status'] == 'Completed'))].index,
    #                    inplace=True)
    #     cash_flow.reset_index(inplace=True, drop=True)
    #     cash_flow['code'] = _get_codes(cash_flow)
    #
    #     return cash_flow

    def get_summed_data(self, year=None, month=None, day=None):
        d = self.get_data(year=year, month=month, day=day)
        if d.shape[0] > 0:
            d = d.loc[:, ['amount', 'code']].groupby('code').sum()
            if 'internal_cashflow' in d.index:
                d.drop(labels='internal_cashflow', inplace=True)
            d.columns = ['total']
        return d

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['amount']
            d.drop(columns=['amount', 'balance'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)
        if range_data.shape[0] > 0:

            withdrawals = range_data[range_data.amount < 0].copy()
            deposits = range_data[range_data.amount >= 0].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'amount') | (columns == 'code')
            withdrawals_ix = (columns == 'amount') | (columns == 'code')
            withdrawals.drop(axis=1, columns=withdrawals.columns[~withdrawals_ix], inplace=True)
            deposits.drop(axis=1, columns=deposits.columns[~deposits_ix], inplace=True)

            withdrawals_counts = withdrawals.groupby(by='code').count()
            deposits_counts = deposits.groupby(by='code').count()

            withdrawals_freqs = np.divide(withdrawals_counts, delta)
            deposits_freqs = np.divide(deposits_counts, delta)

            daily_freqs = withdrawals_freqs.join(other=deposits_freqs, how='outer', lsuffix='_out').replace(np.nan, 0)

            withdrawals_mean = withdrawals.groupby(by='code').mean()
            deposits_mean = deposits.groupby(by='code').mean()

            means = withdrawals_mean.join(other=deposits_mean, how='outer', lsuffix='_out').replace(np.nan, 0)

            withdrawals_std = withdrawals.groupby(by='code').std()
            deposits_std = deposits.groupby(by='code').std()

            std_devs = withdrawals_std.join(other=deposits_std, how='outer', lsuffix='_out').replace(np.nan, 0)

            daily_means = np.multiply(means, daily_freqs)
            daily_std = np.multiply(std_devs, daily_freqs)

            daily_freqs.amount_out *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])
        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

    def update_from_raw_files(self):
        if self.metadata.raw_files_dir == '':
            raise ValueError('please specify a path for Desjardins CSV files')

        csv_files = os.listdir(self.metadata.raw_files_dir)
        cash_flow = pd.DataFrame()
        for x in csv_files:
            if x[-4:].lower() == '.csv':
                df = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
                                 encoding='latin1')
                df.drop(labels=0, inplace=True)
                cash_flow = cash_flow.append(df, ignore_index=True)

        # Convert strings to actual date time objects
        cash_flow.columns = ['date', *cash_flow.columns[1:].str.lower()]
        if 'date' in cash_flow.columns:
            cash_flow['date'] = pd.to_datetime(cash_flow['date'], format='%d/%m/%Y')
        cash_flow.sort_values(by=['date'], inplace=True)

        # Adds column,and inputs transaction code
        cash_flow.insert(loc=5,
                         column='description',
                         value=cash_flow['name'].replace(np.nan, 'NA') + ' - ' + cash_flow['type'])
        cash_flow.drop(columns=['name', 'type'], inplace=True)
        cash_flow = cash_flow.replace(np.nan, 0)

        for ix, row in cash_flow.iterrows():
            if row['status'] == 'Completed' and row['currency'] != 'CAD' and row['balance'] != 0:
                if cash_flow.loc[ix + 1, 'status'] == 'Pending':
                    cash_flow.loc[ix, 'amount'] = -cash_flow.loc[ix + 1, 'amount']
                    cash_flow.loc[ix, 'balance'] = -cash_flow.loc[ix + 1, 'balance']
                else:
                    cash_flow.loc[ix, 'amount'] = np.sign(cash_flow.loc[ix, 'amount']) * abs(
                        cash_flow.loc[ix + 1, 'amount'])
                    cash_flow.loc[ix, 'balance'] = np.sign(cash_flow.loc[ix, 'amount']) * abs(
                        cash_flow.loc[ix + 1, 'balance'])
                cash_flow.loc[ix, 'currency'] = 'CAD'
            elif row['status'] == 'Completed' \
                    and row['description'][-len('Express Checkout Payment'):] == 'Express Checkout Payment' \
                    and row['currency'] == 'CAD' \
                    and row['balance'] == 0:
                cash_flow.loc[ix, 'balance'] = cash_flow.loc[ix, 'amount']
            elif row['status'] == 'Completed' \
                    and (row['description'] == 'NA - General Credit Card Deposit' or
                         row['description'] == 'NA - Reversal of ACH Deposit') \
                    and row['balance'] != 0:
                cash_flow.loc[ix, 'balance'] = 0

        cash_flow.drop(index=cash_flow[~((cash_flow['balance'] != 0) & (cash_flow['status'] == 'Completed'))].index,
                       inplace=True)
        cash_flow.reset_index(inplace=True, drop=True)
        print("this and then some...")
        # TODO fix this, nan being assign after manual code entry
        cash_flow['code'] = get_common_codes(cash_flow)
        l1 = self.transaction_data.shape[0]
        self.transaction_data = self.transaction_data.append(cash_flow)
        self.transaction_data.drop_duplicates(inplace=True)
        self.transaction_data.reset_index(inplace=True, drop=True)
        l2 = self.transaction_data.shape[0]

        if l2 - l1 > 0:
            print(f'{l2 - l1} new entries added')
            if self._save_prompt():
                self.to_pickle()

    # TODO implement prediction method
    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        df = self.get_data()
        return df


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
        for ix, (name, col) in enumerate(data.iteritems()):
            sel = ix % len(line_styles)
            style = line_styles[sel]
            color = (sel * 0.2 % 1.0 + 0.2, random.random(), random.random())
            plt.plot(col, c=color, linestyle=style, label=name)
        # sns.relplot(data=data, kind='line', palette='muted')
        plt.legend()
        plt.xticks(rotation=90)
        plt.show()


"""
Create account objects
"""
# ==============
# Desjardins OP
# ==============
names = ['branch', 'foliot', 'account', 'date', 'transaction_num', 'description', 'fees', 'withdrawal', 'deposit',
         'interests', 'capital_paid', 'advance', 'reimboursment', 'balance']

metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'desjardins_csv_files'),
                           serialized_object_path=os.path.join(pickle_dir, 'desjardins_op.pkl'),
                           planned_transactions_path=os.path.join(data_dir,
                                                                  'desjardins_planned_transactions.csv'),
                           interest_rate=0,
                           name=AccountNames.DESJARDINS_OP,
                           columns_names=names,
                           assignation_file_path=os.path.join(data_dir, 'assignations_desjardins_op.json'),
                           type=AccountType.OPERATIONS,
                           status=AccountStatus.OPEN)
desjardins_op = DesjardinsOP(lmetadata=metadata)

# ==============
# Desjardins MC
# ==============
metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'desjardins_csv_files'),
                           serialized_object_path=os.path.join(pickle_dir, 'desjardins_mc.pkl'),
                           planned_transactions_path=os.path.join(data_dir,
                                                                  'desjardins_planned_transactions.csv'),
                           interest_rate=0.0645,
                           name=AccountNames.DESJARDINS_MC,
                           columns_names=names,
                           assignation_file_path='',
                           type=AccountType.CREDIT,
                           status=AccountStatus.OPEN)
desjardins_mc = DesjardinsMC(lmetadata=metadata)

# ==============
# Desjardins PR
# ==============
metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'desjardins_csv_files'),
                           serialized_object_path=os.path.join(pickle_dir, 'desjardins_pr.pkl'),
                           planned_transactions_path=os.path.join(data_dir,
                                                                  'desjardins_planned_transactions.csv'),
                           interest_rate=0.052 ,
                           name=AccountNames.DESJARDINS_PR,
                           columns_names=names,
                           assignation_file_path='',
                           type=AccountType.CREDIT,
                           status=AccountStatus.OPEN)
desjardins_pr = DesjardinsPR(lmetadata=metadata)

# ==============
# Desjardins PrePaid VISA
# ==============
names = ['acc_name', 'unknown1', 'unknown2',  'date', 'transaction_num', 'description', 'unknown6', 'unknown7',
         'unknown8', 'unknown9', 'unknown10', 'payment/credit',  'credit', 'unknown13']
metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'desjardins_ppcard_csv_files'),
                           serialized_object_path=os.path.join(pickle_dir, 'desjardins_ppcard.pkl'),
                           planned_transactions_path=os.path.join(data_dir,
                                                                  'desjardins_planned_transactions.csv'),
                           interest_rate=0,
                           name=AccountNames.VISA_PP,
                           columns_names=names,
                           assignation_file_path='',
                           type=AccountType.PREPAID,
                           status=AccountStatus.OPEN)
visapp = VisaPP(lmetadata=metadata)

# ==============
# Capital One
# ==============
names = ['date', 'posted_date', 'card_num', 'description', 'category', 'debit', 'credit']
metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'capital_one_csv_files'),
                           serialized_object_path=os.path.join(pickle_dir, 'capital_one.pkl'),
                           planned_transactions_path=os.path.join(data_dir,
                                                                  'desjardins_planned_transactions.csv'),
                           interest_rate=0,
                           name=AccountNames.CAPITAL_ONE,
                           columns_names=names,
                           assignation_file_path='',
                           type=AccountType.CREDIT,
                           status=AccountStatus.CLOSED)
capital_one = CapitalOne(lmetadata=metadata)

# ==============
# CIBC
# ==============
names = ['date', 'description', 'debit', 'credit', 'card_num']
metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'cibc_csv_files'),
                           serialized_object_path=os.path.join(pickle_dir, 'cibc.pkl'),
                           planned_transactions_path=os.path.join(data_dir,
                                                                  'desjardins_planned_transactions.csv'),
                           interest_rate=0,
                           name=AccountNames.CIBC,
                           columns_names=names,
                           assignation_file_path=os.path.join(data_dir, 'assignations_cibc.json'),
                           type=AccountType.CREDIT,
                           status=AccountStatus.OPEN)
cibc = CIBC(lmetadata=metadata)

# ==============
# PayPal
# ==============
names = ['Date', 'Time', 'TimeZone', 'Name', 'description', 'Status', 'Currency', 'Amount', 'Receipt ID', 'Balance']
metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'paypal_csv_files'),
                           serialized_object_path=os.path.join(pickle_dir, 'paypal.pkl'),
                           planned_transactions_path=os.path.join(data_dir,
                                                                  'desjardins_planned_transactions.csv'),
                           interest_rate=0,
                           name=AccountNames.PAYPAL,
                           columns_names=names,
                           assignation_file_path='',
                           type=AccountType.PREPAID,
                           status=AccountStatus.CLOSED)
paypal = PayPal(lmetadata=metadata)

# Create the Accounts object allowing to integrate the data of all the accounts
accounts = Accounts(l_accounts_list=[desjardins_op,
                                     desjardins_mc,
                                     desjardins_pr,
                                     visapp,
                                     capital_one,
                                     cibc,
                                     paypal])
