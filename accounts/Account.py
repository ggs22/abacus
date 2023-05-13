import datetime
import json
import os
import pickle
from abc import ABC, abstractmethod

import colorama
import numpy as np
import pandas as pd

from accounts.Accounts import AccountMetadata, get_common_codes, _barplot_dataframe, print_codes_menu


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