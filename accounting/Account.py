import datetime
import json
import os
import pickle
import random
import re

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
from pathlib import Path

import colorama
import numpy as np
import pandas as pd
import seaborn as sns
import hashlib
from matplotlib import pyplot as plt
from utils import path_utils as pu

from utils.utils import data_dir, months_map
from omegaconf.dictconfig import DictConfig


class AccountNames(Enum):
    DESJARDINS_OP = 1
    DESJARDINS_MC = 2
    DESJARDINS_PR = 3
    VISA_PP = 4
    PAYPAL = 5
    CAPITAL_ONE = 6
    CIBC = 7


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
        g.text(x=(t * (t > 0) + 2), y=i, s=lbl)

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

    common_assignations_path = pu.common_assignations_path
    if not common_assignations_path.exists():
        raise IOError(f"Assignation file not found at {os.path.join(data_dir, 'assignation.json')}")
    with open(common_assignations_path, 'r') as f:
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


def _get_md5(string: str):
    return hashlib.md5(string=string.encode('utf-8')).digest()


class NewRefactoredAccount:

    def __init__(self, conf: DictConfig):

        self.conf = conf
        self.validate_config()

        with open(self.conf.assignation_file_path, 'r') as f:
            self.common_assignations = json.load(fp=f)

        if Path(self.conf.serialized_object_path).exists():
            with open(self.conf.serialized_object_path, 'rb') as f:
                serialized_properties = pickle.load(file=f)
                for prop, value in serialized_properties.__dict__.items():
                    setattr(self, prop, value)
        else:
            # the following objects will be serialized by the self.save() function
            self.processed_data_files = set()
            self.transaction_data: pd.DataFrame = None

        # load raw data
        self._add_from_raw_files()

    def __getstate__(self):
        return {"processed_data_files": self.processed_data_files,
                "transaction_data": self.transaction_data}

    def __setstate__(self, state):
        self.processed_data_files = state["processed_data_files"]
        self.transaction_data = state["transaction_data"]

    def validate_config(self):

        suffix = f"In the yaml configuration file for this Account object {self}, "

        # check that all listed numerical columns are listed in the columns
        for numerical_col in self.conf.numerical_columns:
            if numerical_col[0] not in self.conf.columns_names:
                raise RuntimeError(suffix + "all the numerical_columns must also be listed in columns_names.")

        # check for required columns names
        for required_col_name in ['date', 'description']:
            if required_col_name not in self.conf.columns_names:
                raise RuntimeError(suffix + f"the column name {required_col_name} is required among the columns_names "
                                            f"parameter. Got: {self.conf.columns_names}.")

    def save(self):
        with open(self.conf.serialized_object_path, 'wb') as f:
            pickle.dump(obj=self, file=f)

    def _add_from_raw_files(self) -> None:
        cash_flow = pd.DataFrame(columns=self.conf.columns_names)
        pattern = re.compile(pattern="(conciliation_)?\d{4}[_-]\d{2}([_-]\d{2})?.csv")
        csv_records_files = list()
        for file in Path(pu.accounting_data_dir).joinpath(self.conf.raw_files_dir).glob('*.csv'):
            if pattern.match(file.name):
                csv_records_files.append(file)
        for csv_file in csv_records_files:
            file_hash = _get_md5(csv_file.name)
            if file_hash not in self.processed_data_files:
                self.processed_data_files.add(file_hash)

                new_csv_record = pd.read_csv(filepath_or_buffer=csv_file,
                                             encoding=self.conf.encoding,
                                             names=self.conf.columns_names)

                # Check if there is a filter to apply on the rows:
                if 'rows_selection' in self.conf:
                    drop_index = new_csv_record[
                        new_csv_record[self.conf.rows_selection.filter_column] != self.conf.rows_selection.filter_value
                        ].index
                    new_csv_record.drop(index=drop_index, inplace=True, )
                cash_flow = pd.concat([cash_flow, new_csv_record], ignore_index=True)

            # Convert strings to actual date time objects
            if 'date' not in cash_flow.columns:
                raise ValueError(f"The Account class expects a column named 'date'! Got: {cash_flow.columns}")
            else:
                cash_flow['date'] = pd.to_datetime(cash_flow['date'], format=self.conf.date_format)

            # Adds column,and inputs transaction code
            cash_flow['code'] = get_common_codes(cash_flow)
            na_idx = cash_flow['code'] == 'na'
            cash_flow.loc[na_idx, 'code'] = self.get_account_specific_codes(cashflow=cash_flow[na_idx])
            cash_flow = cash_flow.replace(np.nan, 0)

            if self.transaction_data is not None:
                cash_flow = pd.concat([self.transaction_data, cash_flow], ignore_index=True)

            cash_flow.sort_values(by=['date'], ascending=True, inplace=True)
            cash_flow.reset_index(drop=True, inplace=True)

            cash_flow.reset_index(drop=True, inplace=True)
            self.transaction_data = cash_flow

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

        account_assignations_path = self.conf.assignation_file_path
        if not Path(account_assignations_path).exists():
            raise FileNotFoundError(f"Assignation file not found at {account_assignations_path}")

        with open(account_assignations_path, 'r') as f:
            assignations = json.load(f)

        codes = list()

        # for each transaction entry...
        for index, description in enumerate(descriptions):
            codes.append("na")
            found = False

            # for each transaction code in the assignations list...
            for code, lookup_values in assignations.items():

                # if the assignations are not empty...
                if len(lookup_values) > 0:

                    # for each lookup value...
                    for lookup_value in lookup_values:

                        # check if the lookup value is found in the description.
                        found = description.lower().find(lookup_value.lower()) != -1

                        # if so assign the corresponding code to the trasaction and break
                        if found:
                            codes[len(codes) - 1] = code
                            break
                if found:
                    break
            # if we went through all codes and no assignation was made, prompt user.s
            if not found:
                code_headers = [col_name for col_name, _ in assignations.items()]
                show_menu: bool = True
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
                                  f'Please enter a number between 1 and {len(assignations)}')
                    else:
                        show_menu = False

        return codes


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

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
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


class Accounts:
    def __init__(self, l_accounts_list: List[Account]):
        self.accounts_list: List[Account] = l_accounts_list
        self.accounts_dict: Dict[str] = dict()
        for acc in self.accounts_list:
            self.accounts_dict[acc.metadata.name.name] = acc

    def __iter__(self) -> Account:
        yield from self.accounts_list

    def __len__(self) -> int:
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
                    f'Total: {income + expenses:.2f}({income_avg + expenses_avg:.2f})$'

            return _barplot_dataframe(d=d, title=title, d_avg=d_avg, show=show)

    def barplot_date_range(self, start_date: datetime.date, end_date: datetime.date, show=False):

        d = self.get_summed_data_date_range(start_date=start_date, end_date=end_date)
        d_avg, _, _ = self.get_data_range_daily_average(start_date=start_date, end_date=end_date)

        expenses = d.loc[d['total'] < 0, 'total'].sum()
        income = d.loc[d['total'] >= 0, 'total'].sum()
        if d is not None and d.shape[0] > 0:
            title = f'{start_date} to  {end_date} All accounts\n' + \
                    f'Expenses: {expenses:.2f}$, Income: {income:.2f}$, Total: {income + expenses:.2f}$'
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
