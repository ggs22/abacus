import datetime
import json
import os
import pickle
import random
import re

from copy import deepcopy
from typing import List, Tuple
from pathlib import Path

import colorama
import numpy as np
import pandas as pd
import seaborn as sns
import hashlib
import matplotlib

from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import path_utils as pu

from utils.utils import data_dir
from utils.datetime_utils import months_map, get_period_bounds
from omegaconf.dictconfig import DictConfig


matplotlib.use('TkAgg')

MC_SAMPLING = 'mc_sampling'
PREDICTED_BALANCE = 'predicted_balance'


def _mad(x: pd.Series) -> pd.Series:
    med = x.median()
    res = abs(x - med)
    mad = res.median()
    return mad


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
        self.initial_balance = self.conf.initial_balance
        self.status = self.conf.status

        if self.conf.balance_column is not None:
            self.balance_sign = self.conf.balance_column[1]
        else:
            self.balance_sign = 1

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
    def has_balance_column(self) -> bool:
        return 'balance' in self.transaction_data.columns

    @property
    def numerical_columns(self) -> List[str]:
        return [name for name, _ in self.conf.numerical_columns]

    @property
    def numerical_signs(self) -> List[int]:
        return [sign for _, sign in self.conf.numerical_columns]

    @property
    def positive_columns(self) -> List[str]:
        return [name for name, sign in zip(self.numerical_columns, self.numerical_signs) if sign > 0]

    @property
    def negative_columns(self) -> List[str]:
        return [name for name, sign in zip(self.numerical_columns, self.numerical_signs) if sign < 0]

    @property
    def most_recent_date(self) -> datetime.date:
        return self.transaction_data.tail(n=1).date

    @property
    def current_balance(self) -> float:
        if 'balance' in self.transaction_data.columns:
            bal = float(self.transaction_data.tail(1).balance.to_numpy()[0])
        else:
            names = [col_name for col_name, _ in self.conf.numerical_columns]
            signs = [sign for _, sign in self.conf.numerical_columns]
            bal = (self.transaction_data[names] * signs).sum().sum()
        return np.round(bal, 2)

    def get_balance(self) -> pd.Series:
        if self.has_balance_column:
            balance = self.transaction_data['balance']
        else:
            balance = (self.transaction_data[self.numerical_columns] * self.numerical_signs).cumsum().sum(axis=1)
            balance += self.conf.initial_balance if 'initial_balance' in self.conf else balance
        return balance

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
            t_mad = period_data[['merged', 'code']].groupby(by='code')['merged'].apply(_mad)
            t_std = period_data[['merged', 'code']].groupby(by='code').std(numeric_only=True).merged

            period_stats = pd.DataFrame({'sums': t_sum,
                                         'mean': t_ave,
                                         'median': t_med,
                                         'mad': t_mad,
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

        account_assignations_path = self.conf.assignation_file_path
        if not Path(account_assignations_path).exists():
            raise FileNotFoundError(f"Assignation file not found at {account_assignations_path} "
                                    f"for account {self.name}")

        with open(account_assignations_path, 'r') as f:
            assignations = json.load(f)

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
        self.save()

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

    def get_planned_transactions(self, start_date: str, predicted_days: int = 365) -> pd.DataFrame:
        first_day = datetime.date.fromisoformat(start_date)
        last_day = first_day + datetime.timedelta(days=predicted_days)

        num_days = (last_day - first_day).days
        num_months = int(np.round(num_days/30.5))
        num_years = int(np.round(num_months/12))

        yearly_transactions = self.planned_transactions['recurring']['yearly']
        monthly_transactions = self.planned_transactions['recurring']['monthly']
        unique_transactions = self.planned_transactions['unique']

        planned = {col_name: list() for col_name in self.conf.columns_names}
        planned['code'] = list()
        for description, transactions in yearly_transactions.items():
            for ix in range(num_years + 1):
                amount = transactions[0]
                planned_month = int(transactions[1].split('-')[0])
                planned_date = datetime.date.fromisoformat(
                    f"{str(first_day.year + ix)}-{'0' * (planned_month < 10)}{transactions[1]}"
                )
                if first_day < planned_date < last_day:
                    planned['date'].append(pd.to_datetime(planned_date))
                    if amount <= 0:
                        planned[self.negative_columns[0]].append(amount)
                        planned[self.positive_columns[0]].append(0)
                    elif amount > 0:
                        planned[self.negative_columns[0]].append(0)
                        planned[self.positive_columns[0]].append(amount)
                    planned['description'].append(PREDICTED_BALANCE)
                    planned['code'].append(transactions[2])

        for description, transactions in monthly_transactions.items():
            year_carry_over = 0
            for ix in range(num_months + 1):
                amount = transactions[0]
                planned_day = transactions[1]
                code = transactions[2]
                planned_month = (first_day.month + ix) % 12
                planned_month = 12 if planned_month == 0 else planned_month
                if ix > 0 and planned_month == 1:
                    year_carry_over += 1
                planned_date = datetime.date.fromisoformat(
                    f"{str(first_day.year + year_carry_over)}-{'0' * (planned_month < 10)}{planned_month}-{'0' * (int(transactions[1]) < 10)}{transactions[1]}"
                )
                if first_day < planned_date < last_day:
                    planned['date'].append(pd.to_datetime(planned_date))
                    if amount <= 0:
                        planned[self.negative_columns[0]].append(amount)
                        planned[self.positive_columns[0]].append(0)
                    elif amount > 0:
                        planned[self.negative_columns[0]].append(0)
                        planned[self.positive_columns[0]].append(amount)
                    planned['description'].append(PREDICTED_BALANCE)
                    planned['code'].append(code)

        for description, transactions in unique_transactions.items():
            if isinstance(transactions[0], list):
                unique_transactions = transactions
            else:
                unique_transactions = [transactions]
            for unique_transaction in unique_transactions:
                amount = unique_transaction[0]
                planned_date = datetime.date.fromisoformat(unique_transaction[1])
                code = unique_transaction[2]
                if first_day < planned_date < last_day:
                    if amount <= 0:
                        planned[self.negative_columns[0]].append(amount)
                        planned[self.positive_columns[0]].append(0)
                    elif amount > 0:
                        planned[self.negative_columns[0]].append(0)
                        planned[self.positive_columns[0]].append(amount)

                    planned['date'].append(pd.to_datetime(planned_date))
                    planned['description'].append(PREDICTED_BALANCE)
                    planned['code'].append(code)

        keys_to_pop = list()
        for key, val in planned.items():
            if len(val) == 0:
                keys_to_pop.append(key)
        for key in keys_to_pop:
            planned.pop(key)

        planned = pd.DataFrame(planned)
        if 'date' in planned.columns:
            planned.sort_values(by='date', ascending=True, inplace=True, ignore_index=True)
            planned.reset_index(drop=True, inplace=True)

        return planned

    def get_balance_prediction(self, predicted_days: int = 365, average_over: int = 365) -> pd.DataFrame:
        df: pd.DataFrame = deepcopy(self.transaction_data)
        period_end_date = df.tail(1).date.array.date[0]
        period_start_date = period_end_date - datetime.timedelta(days=average_over)

        stats = self.period_stats(period_start_date.strftime('%Y-%m-%d'),
                                  period_end_date.strftime('%Y-%m-%d'))

        # planned_transactions = self.get_planned_transactions(period_end_date.strftime('%Y-%m-%d'), predicted_days)
        # daily_expense = stats[stats['sums'] < 0].sum(axis=0).sums / average_over
        daily_expense = stats.sum(axis=0).sums / average_over
        pred = {col_name: list() for col_name in df.columns}
        for days_ix, _ in tqdm(enumerate(range(predicted_days))):
            amount = daily_expense
            pred['date'].append(days_ix + 1)
            if amount <= 0:
                pred[self.negative_columns[0]].append(amount)
                pred[self.positive_columns[0]].append(0)
            else:
                pred[self.negative_columns[0]].append(0)
                pred[self.positive_columns[0]].append(amount)
            pred['description'].append(PREDICTED_BALANCE)
            pred['code'].append("other")

        keys_to_pop = list()
        for key, val in pred.items():
            if len(val) == 0:
                keys_to_pop.append(key)
        for key in keys_to_pop:
            pred.pop(key)
        pred = pd.DataFrame(pred)

        if not self.has_balance_column:
            df['balance'] = (df[self.numerical_columns] * self.numerical_signs).cumsum().sum(axis=1)
        df['balance'] += self.initial_balance
        df['balance'] *= self.balance_sign

        current_balance = df.tail(1).balance.item()

        pred['date'] = pd.to_datetime(pred['date'].apply(lambda i: period_end_date + datetime.timedelta(days=i)))

        # pred = pd.concat([pred, planned_transactions], axis=0)
        pred.sort_values(by='date', ascending=True, ignore_index=True, inplace=True)
        pred = pred.replace(np.nan, 0)
        pred['balance'] = pred.loc[:, [self.positive_columns[0], self.negative_columns[0]]].cumsum().sum(axis=1) + current_balance

        pred = pd.concat([df, pred], axis=0)
        pred.reset_index(drop=True, inplace=True)

        return pred

    def get_mc_prediction2(self, predicted_days: int = 365, mc_iterations: int = 500) -> pd.DataFrame:
        # get period data
        df: pd.DataFrame = deepcopy(self.transaction_data)
        period_end_date = df.tail(1).date.array.date[0]
        period_start_date = period_end_date - datetime.timedelta(days=365)
        period_data, days_in_period = self.get_period_data(period_start_date.strftime('%Y-%m-%d'),
                                                           period_end_date.strftime('%Y-%m-%d'))

        stats = self.period_stats(period_start_date.strftime('%Y-%m-%d'),
                                  period_end_date.strftime('%Y-%m-%d'))

        if not self.has_balance_column:
            period_data['balance'] = (period_data[self.numerical_columns] * self.numerical_signs).cumsum().sum(axis=1)
        period_data['balance'] += self.initial_balance
        period_data['balance'] *= self.balance_sign

        current_balance = period_data.tail(1).balance.item()

        pred = {col_name: list() for col_name in period_data.columns}
        for days_ix, _ in tqdm(enumerate(range(predicted_days))):
            for ix, row in stats.iterrows():
                occurence_prob = random.random()
                occured = occurence_prob <= row['daily_prob']
                if occured:
                    amount = random.gauss(mu=row['mean'], sigma=row['mad'])
                    pred['date'].append(days_ix + 1)
                    if amount <= 0:
                        pred[self.negative_columns[0]].append(amount)
                        pred[self.positive_columns[0]].append(0)
                    else:
                        pred[self.negative_columns[0]].append(0)
                        pred[self.positive_columns[0]].append(amount)
                    pred['description'].append(PREDICTED_BALANCE)
                    pred['code'].append(ix)

        keys_to_pop = list()
        for key, val in pred.items():
            if len(val) == 0:
                keys_to_pop.append(key)
        for key in keys_to_pop:
            pred.pop(key)
        pred = pd.DataFrame(pred)

        pred['date'] = pd.to_datetime(pred['date'].apply(lambda i: period_end_date + datetime.timedelta(days=i)))
        pred['balance'] = pred.loc[:, [self.positive_columns[0], self.negative_columns[0]]].cumsum().sum(axis=1) + current_balance
        pred = pred.replace(np.nan, 0)
        pred = pd.concat([period_data, pred], axis=0)
        pred.reset_index(drop=True, inplace=True)

        return pred

    def get_mc_prediction(self, predicted_days: int = 365, mc_iterations: int = 500) -> pd.DataFrame:
        # get period data
        df: pd.DataFrame = deepcopy(self.transaction_data)
        period_end_date = df.tail(1).date.array.date[0]
        period_start_date = period_end_date - datetime.timedelta(days=365)
        period_data, days_in_period = self.get_period_data(period_start_date.strftime('%Y-%m-%d'), period_end_date.strftime('%Y-%m-%d'))

        if not self.has_balance_column:
            period_data['balance'] = (period_data[self.numerical_columns] * self.numerical_signs).cumsum().sum(axis=1)
        period_data['balance'] += self.initial_balance
        period_data['balance'] *= self.balance_sign

        current_balance = period_data.tail(1).balance.item()

        # sample from period data (MC iterations x Num. simulated days) times.
        daily_transaction_frequency = period_data.shape[0] / days_in_period
        sure_transactions: int = int(np.floor(daily_transaction_frequency))
        prob_transactions: float = daily_transaction_frequency % 1

        mc_iterations_list: List[List[pd.DataFrame]] = list()
        for _ in tqdm(range(mc_iterations), desc=f'Sampling {self.name}...'):
            mc_iterations_list.append(list())
            last_ix = len(mc_iterations_list) - 1
            for days_ix, _ in enumerate(range(predicted_days)):
                occurence_prob = random.random()
                occured = occurence_prob < prob_transactions

                num_samples = (sure_transactions + 1) * occured + sure_transactions * (not occured)
                # sample = period_data.sample(n=num_samples)
                sample = period_data.sample(n=num_samples, replace=False)
                sample.date = days_ix + 1
                mc_iterations_list[last_ix].append(sample)

        pred = period_data
        for ix, iteration_list in enumerate(tqdm(mc_iterations_list, desc=f'Processing {self.name}...')):
            df = pd.concat(iteration_list)
            df.loc[:, 'date'] = df.loc[:, 'date'].apply(lambda i: pd.to_datetime(period_end_date + datetime.timedelta(days=i)))
            df.loc[:, 'description'] = MC_SAMPLING
            df.loc[:, 'balance'] = (df[self.numerical_columns] * self.numerical_signs).cumsum().sum(axis=1) + current_balance
            mc_iterations_list[ix] = df

        pred = pd.concat([pred, *mc_iterations_list])
        pred.sort_values(by='date', ascending=True, inplace=True, ignore_index=True)
        pred.reset_index(inplace=True, drop=True)

        return pred

    def plot_mc_prediction(self, predicted_days: int = 365, mc_iterations: int = 500) -> None:
        pred = self.get_mc_prediction(predicted_days=predicted_days, mc_iterations=mc_iterations)
        past_ix = pred.description != MC_SAMPLING
        pred_ix = pred.description == MC_SAMPLING
        pp = pred[pred_ix].groupby(by='date').mean(numeric_only=True)
        plt.plot(pred[past_ix].date, pred[past_ix].balance)
        plt.plot(pp.index, pp.balance)
        plt.title(f"Prediction {predicted_days} days {self.name}")
        plt.xticks(rotation=90)

    def plot_prediction(self, predicted_days: int = 365) -> None:
        pred = self.get_balance_prediction(predicted_days=predicted_days)
        past_ix = pred.description != PREDICTED_BALANCE
        pred_ix = pred.description == PREDICTED_BALANCE
        plt.plot(pred[past_ix].date, pred[past_ix].balance)
        plt.plot(pred[pred_ix].date, pred[pred_ix].balance)
        plt.title(f"Prediction {predicted_days} days {self.name}")
        plt.xticks(rotation=90)

    def histplot(self, period_seed_date: str, date_end: str = ""):
        data, _ = self.get_period_data(period_seed_date=period_seed_date, date_end=date_end)
        if data is not None:
            cols = list()
            signs = list()
            for col, sign in self.conf.numerical_columns:
                cols += [col]
                signs += [sign]
            data['merged'] = (data.loc[:, cols] * signs).sum(axis=1)
            sns.histplot(data=data, x='merged', log_scale=(False, True))
            plt.title(f"{self.name}\n{period_seed_date}" + f" - {date_end}" * (date_end != ""))

    def barplot(self, period_seed_date: str, date_end: str = ""):
        data, _ = self.get_period_data(period_seed_date=period_seed_date, date_end=date_end)
        if data is not None:
            cols = list()
            signs = list()
            for col, sign in self.conf.numerical_columns:
                cols.append(col)
                signs.append(sign)
            data = data.groupby(by='code').sum(numeric_only=True)
            data['total'] = (data.loc[:, cols] * signs).sum(axis=1)
            plt.figure(num=f"{self.name} - {period_seed_date}")
            plt.title(label=f"{self.name} - {period_seed_date}")
            sns.barplot(data, x='total', y=data.index)


