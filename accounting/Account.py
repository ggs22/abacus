import datetime
import json
import os
import pickle
import re

from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import colorama
import numpy as np
import pandas as pd
import seaborn as sns
import hashlib

from matplotlib import pyplot as plt
from utils import path_utils as pu

from utils.utils import data_dir
from utils.datetime_utils import months_map, get_period_bounds
from omegaconf.dictconfig import DictConfig


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
                        "common_assignations.csv"
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


def _get_md5(string: str):
    return hashlib.md5(string=string.encode('utf-8')).digest()


class Account:

    def __init__(self, conf: DictConfig):

        self.conf = conf
        self.validate_config()

        self.name = self.conf.name

        with open(self.conf.assignation_file_path, 'r') as f:
            self.common_assignations = json.load(fp=f)

        with open(self.conf.planned_transactions_path, 'r') as f:
            self.planned_transactions = json.load(fp=f)

        # If serialized objects exist, load them.
        if Path(self.conf.serialized_object_path).exists():
            with open(self.conf.serialized_object_path, 'rb') as f:
                serialized_properties = pickle.load(file=f)
                for prop, value in serialized_properties.__dict__.items():
                    setattr(self, prop, value)
        # Otherwise initialize properties from scratch.
        else:
            # the following objects will be serialized by the self.save() function
            self.processed_data_files = set()
            self.transaction_data: pd.DataFrame = None

        # load raw data
        self._import_csv_files()

    def __getstate__(self):
        return {"processed_data_files": self.processed_data_files,
                "transaction_data": self.transaction_data}

    def __setstate__(self, state):
        self.processed_data_files = state["processed_data_files"]
        self.transaction_data = state["transaction_data"]

    def __getitem__(self, item):
        ret = self.transaction_data.loc[item, :]
        return ret

    @property
    def most_recent_date(self) -> datetime.date:
        return self.transaction_data.tail(n=1).date

    @property
    def balance(self) -> float:
        if 'balance' in self.transaction_data.columns:
            bal = float(self.transaction_data.tail(1).balance.to_numpy()[0])
        else:
            names = [col_name for col_name, _ in self.conf.numerical_columns]
            signs = [sign for _, sign in self.conf.numerical_columns]
            bal = (self.transaction_data[names] * signs).sum().sum()
        return np.round(bal, 2)

    def get_period_data(self, period_seed_date: str, date_end: str = "") -> Tuple[pd.DataFrame, int]:
        first_day, last_day = get_period_bounds(period_seed_date, date_end)
        period_data = deepcopy(self.transaction_data)
        period_data = period_data[(period_data['date'].array.date >= first_day) &
                                  (period_data['date'].array.date <= last_day)]
        if len(period_data) > 0:
            # The data is supposed to be sorted in ascending date order
            first_day = period_data.head(1)['date'].array.date[0]
            last_day = period_data.tail(1)['date'].array.date[0]
            days = (last_day - first_day).days + 1
        else:
            period_data, days = None, 0

        return period_data, days

    def period_stats(self, date: str, date_end: str = "") -> pd.DataFrame:
        period_data, delta_days = self.get_period_data(date, date_end)

        if period_data is not None:
            num_cols = [col for col, _ in self.conf.numerical_columns]
            signs = [sign for _, sign in self.conf.numerical_columns]

            period_data[num_cols] *= signs
            period_data['merged'] = period_data[num_cols].sum(axis=1)
            period_data.drop(columns=num_cols, inplace=True)

            t_counts = period_data[['merged', 'code']].groupby(by='code').count().merged
            t_daily_prob = (t_counts / delta_days)

            t_sum = period_data[['merged', 'code']].groupby(by='code').sum(numeric_only=True).merged
            t_ave = period_data[['merged', 'code']].groupby(by='code').mean(numeric_only=True).merged
            t_med = period_data[['merged', 'code']].groupby(by='code').median(numeric_only=True).merged
            t_std = period_data[['merged', 'code']].groupby(by='code').std(numeric_only=True).merged

            period_stats = pd.DataFrame({'sums': t_sum,
                                         'mean': t_ave,
                                         'median': t_med,
                                         'std': t_std,
                                         'daily_prob': t_daily_prob,
                                         'count': t_counts},
                                        index=t_std.index).replace(np.nan, 0)
        else:
            period_stats = None

        return period_stats

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

    def _import_csv_files(self) -> None:
        pattern = re.compile(pattern="(conciliation_)?\d{4}[_-]\d{2}([_-]\d{2})?.csv")
        csv_records_files = list()
        for file in Path(pu.accounting_data_dir).joinpath(self.conf.raw_files_dir).glob('*.csv'):
            if pattern.match(file.name):
                csv_records_files.append(file)
        for csv_file in csv_records_files:
            file_hash = _get_md5(csv_file.name)
            if file_hash not in self.processed_data_files:
                print(f"Importing new data from {csv_file.name}")

                cash_flow = pd.read_csv(filepath_or_buffer=csv_file,
                                        encoding=self.conf.encoding,
                                        names=self.conf.columns_names,
                                        header=0)

                # Check if there is a filter to apply on the rows:
                if self.conf.rows_selection is not None:
                    drop_index = cash_flow[
                        cash_flow[self.conf.rows_selection.filter_column] != self.conf.rows_selection.filter_value
                        ].index
                    cash_flow.drop(index=drop_index, inplace=True, )

                # Make sure all imported data types are coherent
                if 'date' not in cash_flow.columns:
                    raise ValueError(f"The Account class expects a column named 'date'! Got: {cash_flow.columns}")
                else:
                    cash_flow['date'] = pd.to_datetime(cash_flow['date'], format=self.conf.date_format)

                for numerical_column in self.conf.numerical_columns:
                    col_name, col_sign = numerical_column[0], numerical_column[1]
                    cash_flow[col_name] = pd.to_numeric(cash_flow[col_name])

                # Automatically assign commone transaction code
                cash_flow['code'] = get_common_codes(cash_flow).to_numpy()
                na_idx = cash_flow['code'] == 'na'
                cash_flow.loc[na_idx, 'code'] = self._get_account_specific_codes(cash_flow[na_idx])
                cash_flow = cash_flow.replace(np.nan, 0)

                if self.transaction_data is not None:
                    cash_flow = pd.concat([self.transaction_data, cash_flow], ignore_index=True)

                cash_flow.drop_duplicates(subset=cash_flow.columns[~(cash_flow.columns == 'code')], inplace=True)
                cash_flow.sort_values(by=list(self.conf.sorting_order), ascending=True, inplace=True)
                cash_flow.reset_index(drop=True, inplace=True)

                self.transaction_data = cash_flow
                self.processed_data_files.add(file_hash)

    def interactive_codes_update(self) -> None:
        na_idx = self.transaction_data.code == 'na'
        cashflow = self.transaction_data[na_idx]
        # descriptions = cashflow.loc[:, 'description']

        account_assignations_path = self.conf.assignation_file_path
        if not Path(account_assignations_path).exists():
            raise FileNotFoundError(f"Assignation file not found at {account_assignations_path} "
                                    f"for account {self.name}")

        with open(account_assignations_path, 'r') as f:
            assignations = json.load(f)

        codes = list()

        for row in self.transaction_data[na_idx].itertuples():
            code_headers = [col_name for col_name, _ in assignations.items()]
            show_menu: bool = True
            while show_menu:
                print_codes_menu(code_headers, self.transaction_data.iloc[row.Index].dropna().to_string())
                code = input()
                if code != 'na':
                    try:
                        code = int(code)
                        if code <= 0 or code > len(code_headers):
                            print(f'{colorama.Fore.RED}Error: {colorama.Fore.RESET}'
                                  f'Please enter a number between 1 and {len(code_headers)}')
                        else:
                            code = code_headers[code - 1]
                            self.transaction_data.loc[row.Index, 'code'] = code
                            show_menu = False
                    except ValueError:
                        print(f'{colorama.Fore.RED}Error: {colorama.Fore.RESET}'
                              f'Please enter a number between 1 and {len(assignations)}')
                else:
                    show_menu = False

        self.transaction_data.loc[na_idx, 'code'] = codes

    def _get_account_specific_codes(self, cashflow: pd.DataFrame, description_column="description") -> List[str]:
        """
        This function returns a vector corresponding to all transaction code associated to the description vector given
        as argument

        args:
            - description   A vector of transaction description (often reffered to as "object, item, description, etc."
                            in bank statements). The relations between the descriptions and the codes are contained in
                            "assignations_<account_name>.csv"
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
        return codes

    def change_transaction_code(self, ix, code):
        self.transaction_data.loc[ix, 'code'] = code

    def clear_period(self, period_seed_date: str, date_end: str = "", inplace=False) -> pd.DataFrame:
        period_data, _ = self.get_period_data(period_seed_date, date_end)
        if period_data is not None:
            ix = period_data.index
            cleared_period = self.transaction_data.drop(labels=ix, inplace=inplace)
        else:
            cleared_period = None
        return cleared_period
