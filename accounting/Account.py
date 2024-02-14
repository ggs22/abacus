import datetime
import json
import os
import pickle
import logging

from copy import deepcopy
from typing import List, Tuple, Sequence
from pathlib import Path

import colorama
import numpy as np
import pandas as pd
import seaborn as sns
import hashlib
import matplotlib

from matplotlib import pyplot as plt
from utils import path_utils as pu

from utils.utils import data_dir
from utils.datetime_utils import get_period_bounds
from omegaconf.dictconfig import DictConfig


matplotlib.use('TkAgg')

MC_SAMPLING = 'mc_sampling'
PREDICTED_BALANCE = 'predicted_balance'

AccountStats = pd.DataFrame
TransactionData = pd.DataFrame


def _mad(x: pd.Series) -> pd.Series:
    med = x.median()
    res = abs(x - med)
    mad = res.median()
    return mad


def print_codes_menu(codes: Sequence[str], transaction: pd.Series):
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


def get_common_codes(cashflow: TransactionData, description_column="description") -> pd.Series:
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

        self.name = self.conf.name

        self.validate_config()

        self.account_dir = conf.account_dir
        self.initial_balance = self.conf.initial_balance
        self.status = self.conf.status

        self.ignored_index: List[int] = None

        self.prediction = None
        self.legacy_account = None
        self.use_legacy_stats = False

        self.logger = logging.getLogger(f"{self.__class__.__name__} - {self.name}")

        if self.conf.balance_column is not None:
            self.balance_sign = self.conf.balance_column[1]
        else:
            self.balance_sign = 1

        with open(Path(self.account_dir).joinpath("planned_transactions.json"), 'r') as f:
            self.planned_transactions = json.load(fp=f)

        # If serialized objects exist, load them.
        if self.serialized_self_path.exists():
            logger = logging.getLogger(self.logger_name)
            with open(self.serialized_self_path, 'rb') as f:
                serialized_properties = pickle.load(file=f)
                for prop, value in serialized_properties.__dict__.items():
                    setattr(self, prop, value)
                logger.info(f"loaded from {self.serialized_self_path}")
        # Otherwise initialize properties from scratch.
        else:
            # the following objects will be serialized by the self.save() function
            self.processed_data_files = set()
            self.transaction_data: pd.DataFrame = None

        # load raw data
        self._import_csv_files()

    def __repr__(self):
        pres = f"{self.name}\n"
        pres += f"{len(self.transaction_data)} entries\n"
        pres += f"last entry: {self.most_recent_date}.\n"
        pres += f"balance: {self.current_balance:.2f}$\n"
        pres += f"type: {self.conf.type}\n"
        pres += f"status: {self.conf.status}\n"
        return pres

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
    def logger_name(self) -> str:
        return f"Account_{self.name}"

    @property
    def serialized_self_path(self) -> Path:
        return Path(self.account_dir).joinpath("pickle_objects", f"{self.name}.pkl")

    @property
    def balance_column(self) -> pd.DataFrame:
        if not self.has_balance_column:
            balance_column = deepcopy((self.transaction_data[self.numerical_names] * self.numerical_signs))
            balance_column = balance_column.cumsum().sum(axis=1)
        else:
            balance_column = deepcopy(self.transaction_data[self.conf.balance_column[0]])
            balance_column.columns = None

        balance_column += self.initial_balance
        balance_column *= self.balance_sign
        balance_column = pd.DataFrame(balance_column)
        balance_column[f"balance_{self.name}"] = balance_column[balance_column.columns[0]]
        balance_column.index = self.transaction_data['date']
        balance_column.drop(columns=balance_column.columns[0], inplace=True)

        return balance_column

    @property
    def has_balance_column(self) -> bool:
        balance_in_conf = self.conf.balance_column is not None
        balance_in_headers = False if not balance_in_conf else self.conf.balance_column[0] in self.transaction_data.columns
        return balance_in_conf and balance_in_headers

    @property
    def numerical_names(self) -> List[str]:
        return [name for name, _ in self.conf.numerical_columns]

    @property
    def numerical_signs(self) -> List[int]:
        return [sign for _, sign in self.conf.numerical_columns]

    @property
    def positive_names(self) -> List[str]:
        return [name for name, sign in zip(self.numerical_names, self.numerical_signs) if sign > 0]

    @property
    def negative_names(self) -> List[str]:
        return [name for name, sign in zip(self.numerical_names, self.numerical_signs) if sign < 0]

    @property
    def columns_names(self) -> List[str]:
        return self.transaction_data.columns

    @property
    def most_recent_date(self) -> datetime.date:
        return self.transaction_data.tail(n=1).date

    @property
    def current_balance(self) -> float:
        return self.balance_column.iloc[-1:, 0].item()

    def _remove_csv_file_from_records(self, csv_file: Path | str) -> None:
        csv_file = Path(csv_file)
        file_hash = _get_md5(csv_file.name)
        try:
            self.processed_data_files.remove(file_hash)
        except IndexError as e:
            self.logger.error(f"Couldn't remove {str(csv_file)} from {self.name} records!:\n{e}")

    def _import_csv_files(self) -> None:
        csv_records_files = list()
        for file in Path(self.account_dir).joinpath("csv_data").glob('*.csv'):
            csv_records_files.append(file)
        for csv_file in csv_records_files:
            file_hash = _get_md5(csv_file.name)
            if file_hash not in self.processed_data_files:
                self.logger.info(f"Importing new data from {csv_file.name} for {self.name}")

                cash_flow = pd.read_csv(filepath_or_buffer=csv_file,
                                        encoding=self.conf.encoding,
                                        names=self.conf.columns_names,
                                        sep=self.conf.separator,
                                        skip_blank_lines=True,
                                        header=0 if self.conf.has_header_row else None)

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

                cash_flow.drop_duplicates(subset=cash_flow.columns[~(cash_flow.columns == 'code')],
                                          inplace=True)
                cash_flow.sort_values(by=list(self.conf.sorting_order),
                                      ascending=list(self.conf.sorting_ascendence),
                                      inplace=True)
                cash_flow.reset_index(drop=True, inplace=True)

                self.transaction_data = cash_flow
                self.processed_data_files.add(file_hash)

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

        account_assignations_path = Path(self.account_dir).joinpath("assignations.json").resolve()
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

    def get_balance(self) -> pd.Series:
        if self.has_balance_column:
            balance = self.transaction_data[self.conf.balance_column[0]]
        else:
            balance = (self.transaction_data[self.numerical_names] * self.numerical_signs).cumsum().sum(axis=1)
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

    def period_stats(self, date: str, date_end: str = "", **kwargs) -> AccountStats | None:

        if self.legacy_account is not None and self.use_legacy_stats:
            # TODO potential for endless recursion, re-think this
            period_stats = self.legacy_account.period_stats(date, date_end, **kwargs)
        else:
            period_data, delta_days = self.get_period_data(date, date_end)

            if period_data is not None:

                if self.ignored_index is not None:
                    period_data = period_data[~period_data.index.isin(self.ignored_index)]

                num_cols = [col for col, _ in self.conf.numerical_columns]
                signs = [sign for _, sign in self.conf.numerical_columns]

                period_data[num_cols] *= signs
                period_data['merged'] = period_data[num_cols].sum(axis=1)
                period_data.drop(columns=num_cols, inplace=True)

                t_counts = period_data[['merged', 'code']].groupby(by='code').count().merged
                t_daily_prob = (t_counts / delta_days)

                t_sum = period_data[['merged', 'code']].groupby(by='code').sum(numeric_only=True).merged
                t_ave = period_data[['merged', 'code']].groupby(by='code').mean(numeric_only=True).merged
                t_daily_ave = t_sum/delta_days
                t_med = period_data[['merged', 'code']].groupby(by='code').median(numeric_only=True).merged

                t_daily_std = dict()
                for code in t_daily_ave.index:
                    _d = period_data[period_data['code'] == code]['merged'].to_numpy()
                    t_daily_std[code] = np.sqrt(np.sum(np.power(_d - t_daily_ave[code].item(), 2))/delta_days)
                t_daily_std = pd.Series(t_daily_std)

                t_mad = period_data[['merged', 'code']].groupby(by='code')['merged'].apply(_mad)
                t_std = period_data[['merged', 'code']].groupby(by='code').std(numeric_only=True).merged

                period_stats = pd.DataFrame({'sums': t_sum,
                                             'daily_mean': t_daily_ave,
                                             'mean': t_ave,
                                             'transac_median': t_med,
                                             'daily_std': t_daily_std,
                                             'mad': t_mad,
                                             'std': t_std,
                                             'daily_prob': t_daily_prob,
                                             'count': t_counts},
                                            index=t_std.index).replace(np.nan, 0)
            else:
                period_stats = None

        return period_stats

    def validate_config(self):

        suffix = f"In the yaml configuration file for the Account, "

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
        destination = self.serialized_self_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(str(destination), 'wb') as f:
            pickle.dump(obj=self, file=f)
            logger = logging.getLogger(self.logger_name)
            logger.info(f"Saved to {str(destination)}")

    def export(self):
        export_dir = Path(self.account_dir).joinpath('exports')
        export_dir.mkdir(parents=True, exist_ok=True)
        with open(export_dir.joinpath(f"transaction_data_{self.name}.csv"), 'w') as transaction_export:
            self.transaction_data.to_csv(transaction_export, sep="\t")
        with open(export_dir.joinpath(f"processed_files_{self.name}.pkl"), 'wb') as processed_files_export:
            pickle.dump(self.processed_data_files, processed_files_export)
        self.logger.info(f"exported transaction data & processed files list in {str(export_dir)}")

    def interactive_codes_update(self) -> None:
        na_idx = self.transaction_data.code == 'na'

        if sum(na_idx) > 0:
            account_assignations_path = Path(self.account_dir).joinpath("assignations.json").resolve()
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
                                self.logger.error(f'{colorama.Fore.RED}Error: {colorama.Fore.RESET}'
                                                  f'Please enter a number between 1 and {len(code_headers)}')
                            else:
                                code = code_headers[code - 1]
                                self.transaction_data.loc[row.Index, 'code'] = code
                                show_menu = False
                        except ValueError:
                            self.logger.error(f'{colorama.Fore.RED}Error: {colorama.Fore.RESET}'
                                              f'Please enter a number between 1 and {len(assignations)}')
                    else:
                        show_menu = False

            ans = input(f"Save {self.name}? ([y]/n):\n")
            if ans in ["", "y", "Y"]:
                self.save()

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

    def get_planned_transactions(self, start_date: str, predicted_days: int = 365) -> pd.DataFrame | None:
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
                    planned[self.negative_names[0]].append(amount * (amount <= 0))
                    planned[self.positive_names[0]].append(amount * (not (amount <= 0)))
                    planned['description'].append(PREDICTED_BALANCE)
                    planned['code'].append(transactions[2])

        for description, transactions in monthly_transactions.items():
            year_carry_over = 0
            for ix in range(num_months + 1):
                amount = transactions[0]
                code = transactions[2]
                planned_month = (first_day.month + ix) % 12
                planned_month = 12 if planned_month == 0 else planned_month
                if ix > 0 and planned_month == 1:
                    year_carry_over += 1
                planned_date = datetime.date.fromisoformat(
                    f"{str(first_day.year + year_carry_over)}-"
                    f"{'0' * (planned_month < 10)}{planned_month}-"
                    f"{'0' * (int(transactions[1]) < 10)}{transactions[1]}"
                )
                if first_day < planned_date < last_day:
                    planned['date'].append(pd.to_datetime(planned_date))
                    planned[self.negative_names[0]].append(amount * (amount <= 0))
                    planned[self.positive_names[0]].append(amount * (not (amount <= 0)))
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
                        planned[self.negative_names[0]].append(amount)
                        planned[self.positive_names[0]].append(0)
                    elif amount > 0:
                        planned[self.negative_names[0]].append(0)
                        planned[self.positive_names[0]].append(amount)

                    planned['date'].append(pd.to_datetime(planned_date))
                    planned['description'].append(PREDICTED_BALANCE)
                    planned['code'].append(code)

        keys_to_pop = list()
        for key, val in planned.items():
            if len(val) == 0:
                keys_to_pop.append(key)
        for key in keys_to_pop:
            planned.pop(key)

        if len(planned) > 0:
            planned = pd.DataFrame(planned)
            if 'date' in planned.columns:
                planned.sort_values(by='date', ascending=True, inplace=True, ignore_index=True)
                planned.reset_index(drop=True, inplace=True)
        else:
            self.logger.info(f"No transactions planned for {self.name}")
            planned = None
        return planned

    def plot(self, figure_name: str = None, c: str | None = None):
        plt.figure(num=figure_name)
        plt.plot(self.balance_column, label=self.name, c=c)

    def histplot(self, period_seed_date: str, date_end: str = "", c: str | None = None):
        data, _ = self.get_period_data(period_seed_date=period_seed_date, date_end=date_end)
        if data is not None:
            plt.figure(f"Histplot {self.name} - {period_seed_date}" + f" {date_end}" * (date_end != ""))
            cols = list()
            signs = list()
            for col, sign in self.conf.numerical_columns:
                cols += [col]
                signs += [sign]
            data['merged'] = (data.loc[:, cols] * signs).sum(axis=1)
            sns.histplot(data=data, x='merged', log_scale=(False, False), color=c)
            plt.title(f"{self.name}\n{period_seed_date}" + f" - {date_end}" * (date_end != ""))

    def barplot(self, period_seed_date: str, date_end: str = ""):
        data, period_length = self.get_period_data(period_seed_date=period_seed_date, date_end=date_end)
        if data is not None:
            data: pd.DataFrame = data.groupby(by='code').sum(numeric_only=True)
            data['total'] = (data.loc[:, self.numerical_names] * self.numerical_signs).sum(axis=1)
            data.sort_values(by='total', ascending=False, inplace=True)
            plt.figure(num=f"{self.name} - {period_seed_date}")

            income = data[data['total'] > 0]['total'].sum()
            expenses = data[data['total'] <= 0]['total'].sum()
            plt.title(label=f"{self.name} - {period_seed_date}\n"
                            f"in: {income: .2f}, out: {expenses: .2f}, bal: {income+expenses: .2f}")
            colors = list()
            for amount in data.total:
                color = (.75, 0, 0) if amount <= 0 else (0, .75, 0)
                colors.append(color)
            plt.barh(y=data.index, width=data.total, color=colors)
            for stat in data.index:
                plt.text(x=max(0, data.loc[stat, 'total']) + 10,
                         y=stat,
                         s=f"{data.loc[stat, 'total']: .2f} / {data.loc[stat, 'total']/period_length: .2f}")

    def filter_by_code(self, code: str, period_seed_date: str = "", date_end: str = "") -> pd.DataFrame | None:
        if period_seed_date == "":
            filtered_data = deepcopy(self.transaction_data)
        else:
            filtered_data, _ = self.get_period_data(period_seed_date=period_seed_date,
                                                    date_end=date_end)

        if filtered_data is not None:
            filtered_data = filtered_data[filtered_data['code'] == code]
            filtered_data = filtered_data if not filtered_data.empty else None

        return filtered_data

    def filter_by_description(self, description: str, period_seed_date: str = "", date_end: str = "") -> pd.DataFrame | None:
        if period_seed_date == "":
            filtered_data = deepcopy(self.transaction_data)
        else:
            filtered_data, _ = self.get_period_data(period_seed_date=period_seed_date,
                                                    date_end=date_end)

        if filtered_data is not None:
            filtered_data = filtered_data[filtered_data["description"].str.contains(description)]
            filtered_data = filtered_data if not filtered_data.empty else None

        return filtered_data