import matplotlib.pyplot as plt
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


class AccountNames(Enum):
    DESJARDINS_OP = 1
    DESJARDINS_MC = 2
    DESJARDINS_PR = 3
    VISA_PP = 4
    PAYPAL = 5
    CAPITAL_ONE = 6


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
    # unique_keys: list
    type: AccountType
    status: AccountStatus


def _barplot_dataframe(d: pd.DataFrame, title: str, figsize=(7, 7), show=False):
    d = d.sort_values(by='total')
    fig = plt.figure(figsize=figsize)
    clrs = ['green' if (x > 0) else 'red' for x in d['total']]
    g = sns.barplot(x='total', y=d.index, data=d, palette=clrs, tick_label=str(d['total'].values))
    for i, t in enumerate(d['total']):
        g.text(x=t, y=i, s=f'{t:.2f}')

    plt.title(title)
    plt.subplots_adjust(left=0.25)
    if show:
        plt.show()
    return fig


def _print_codes_menu(codes, transaction):
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


def _get_codes(cashflow: pd.DataFrame, description_column="description") -> pd.Series:
    """
    This function returns a vector corresponding to all transaction code associated to the description vector given
    as argument

    args:
        - description   A vector of transaction description (often reffered to as "object, item, description, etc."
                        in bank statements). The relations between the descriptions and the codes are contained in
                        "assignations.csv"
    """

    descriptions = cashflow.loc[:, description_column]

    assignations = pd.read_csv(os.path.join(data_dir, 'assignations.csv'), encoding='utf-8', sep=',').dropna(axis=1,
                                                                                                             how='all')
    codes = list()
    for index, description in enumerate(descriptions):
        codes.append("na")
        for col in assignations.iteritems():
            for row in col[1].dropna(axis=0):
                if description.lower().find(row.lower()) != -1:
                    codes[len(codes) - 1] = col[0]
                    break
        if codes[-1:] == ['na']:
            show_menu = True
            while show_menu:
                _print_codes_menu(assignations.columns, cashflow.iloc[index].dropna().to_string())
                code = input()
                if code != 'na':
                    try:
                        code = int(code)
                        if code <= 0 or code > len(assignations.columns.values):
                            print(f'{colorama.Fore.RED}Error: {colorama.Fore.RESET}'
                                  f'Please enter a number between 1 and {len(assignations.columns)}')
                        else:
                            code = assignations.columns.values[code - 1]
                            codes[-1:] = [code]
                            show_menu = False
                    except ValueError:
                        print(f'{colorama.Fore.RED}Error: {colorama.Fore.RESET}'
                              f'Please enter a number between 1 and {len(assignations.columns)}')
                else:
                    show_menu = False

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
        return d.groupby(by='code').sum()

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        if self.col_mask is not None:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date).loc[:, self.col_mask]
        else:
            d = self.get_data_by_date_range(start_date=start_date, end_date=end_date)
        return d.groupby(by='code').sum()

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

        d = self.get_data_by_date(year=year, month=month, day=day)

        if d.shape[0] > 0:
            sdate = d.date.head(1).array.date
            edate = d.date.tail(1).array.date
            delta = (edate - sdate)[0].days
            summed_data = self.get_summed_data(year=year, month=month, day=day)
            if delta == 0:
                res = summed_data
            else:
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

        d = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if d.shape[0] > 0:
            delta = (end_date - start_date).days
            summed_data = self.get_summed_date_range_data(start_date=start_date, end_date=end_date)
            if delta == 0:
                res = summed_data
            else:
                res = np.divide(summed_data, delta)
        else:
            res = None

        return res

    def get_data_by_code(self, code: str, year=None, month=None, day=None):
        d = self.get_data(year=year, month=month, day=day)
        return d.loc[d['code'] == code]

    def _load_from_raw_files(self, update_data=False) -> None:
        # TODO keep a record of processed raw files as to avoid reprocessing them in updates
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
            cash_flow['code'] = _get_codes(cash_flow)
            cash_flow = cash_flow.replace(np.nan, 0)
            cash_flow.drop_duplicates(subset=cash_flow.columns[cash_flow.columns != 'code'])

            df = cash_flow

        if df is None:
            raise FileNotFoundError(f'No serialized or raw data (in {self.metadata.raw_files_dir}) was found for '
                                    f'{self.metadata.name}')
        else:
            self.transaction_data = df

        # if self.metadata.raw_files_dir == '':
        #     raise ValueError('please specify a path for Desjardins CSV files')
        #
        # csv_files = os.listdir(self.metadata.raw_files_dir)
        # if self.transaction_data is None:
        #     cash_flow = pd.DataFrame(columns=self.metadata.columns_names)
        # else:
        #     cash_flow = self.transaction_data
        # for x in csv_files:
        #     if x[-4:] == '.csv':
        #         cash_flow = cash_flow.append(
        #             pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
        #                         encoding='latin1',
        #                         names=self.metadata.columns_names),
        #             ignore_index=True)
        #
        # # Convert strings to actual date time objects
        # if 'date' in cash_flow.columns:
        #     cash_flow['date'] = pd.to_datetime(cash_flow['date'], format='%Y-%m-%d')
        #
        # # Adds column,and inputs transaction code
        # cash_flow['code'] = _get_codes(cash_flow)
        # cash_flow = cash_flow.replace(np.nan, 0)
        #
        # return cash_flow

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

    def barplot(self, year=None, month=None, day=None, show=False, average: bool = False):

        if not average:
            d = self.get_summed_data(year=year, month=month, day=day)
        else:
            d = self.get_daily_average(year=year, month=month, day=day)

        if d is not None and d.shape[0] > 0:
            title = f'{year} ' * (year is not None) + \
                    f'{month} ' * (month is not None) + \
                    f'{day} ' * (day is not None) + f'{self.get_name()}'
            return _barplot_dataframe(d=d, title=title, show=show)

    def get_name(self):
        s = str(self.metadata.name).split('.')[1]
        s = s.replace('_', ' ')
        return s

    def change_transaction_code(self, ix, code):
        self.transaction_data.loc[ix, 'code'] = code

    def apply_description_filter(self, pattern: str, regex=False):
        ret = self.transaction_data.loc[
              self.transaction_data["description"].str.contains("Paiement /PAYPAL", regex=True), :]
        return ret

    @abstractmethod
    def update_from_raw_files(self):
        """Each account type has its own update method"""

    @abstractmethod
    def get_predicted_balance(self):
        """Each account type has its own prediction method"""


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

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_date_range_daily_average(start_date=start_date, end_date=end_date)
        d.drop(labels=['internal_cashflow'], inplace=True)
        d['balance'] = np.subtract(d['deposit'], d['withdrawal'])
        d.drop(columns=['transaction_num', 'withdrawal', 'deposit'], inplace=True)
        return d

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
                        if row['account'] == 'EOP':
                            row['date'] = pd.to_datetime(row['date'], format='%Y-%m-%d')
                            row['code'] = _get_codes(pd.DataFrame(row).T).values[0]
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

    # def barplot(self, year=None, month=None, day=None, show=False, figsize=(7, 7), average: bool = False):
    #
    #     if not average:
    #         d = self.get_summed_data(year=year, month=month, day=day)
    #     else:
    #         d = self.get_summed_average(year=year, month=month, day=day)
    #
    #     fig = plt.figure(figsize=figsize)
    #     clrs = ['green' if (x > 0) else 'red' for x in d['total']]
    #     g = sns.barplot(x='total', y=d.index, data=d, palette=clrs, tick_label=str(d['total'].values))
    #     for i, t in enumerate(d['total']):
    #         g.text(x=t, y=i, s=f'{t:.2f}')
    #     plt.title(f'{year}-{month}')
    #     plt.subplots_adjust(left=0.25)
    #     if show:
    #         plt.show()
    #     return fig


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

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_date_range_daily_average(start_date=start_date, end_date=end_date)
        d.drop(labels=['internal_cashflow'], inplace=True)
        d['balance'] = np.subtract(d['reimboursment'], d['advance'])
        d['balance'] = np.subtract(d['balance'], d['interests'])
        d.drop(columns=['transaction_num', 'advance', 'reimboursment', 'interests'], inplace=True)
        return d

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
                            row['code'] = _get_codes(pd.DataFrame(row).T).values[0]
                            row.replace(np.nan, 0, inplace=True)
                            self.transaction_data = self.transaction_data.append(other=row, ignore_index=True)
                            new_entries += 1

        if new_entries > 0:
            self.transaction_data.drop_duplicates(keep='first', subset=self.metadata.columns_names, inplace=True)
            self.transaction_data.sort_values(by=['date', 'transaction_num'], inplace=True)

            print(f'{new_entries} new entries added')
            if self._save_prompt():
                self.to_pickle()

    def get_predicted_balance(self, days: int = 91, sim_date: datetime.date = None) -> pd.DataFrame:
        """
        This function gives an estimation of the future balance of the account for a specified number of days. It
        uses the planned transaction data and optionally the average spending data.
        :param days: number of days predicted
        :param sim_date: date at which the prediction start
        :return: a Dataframe containing the predicted balance of the account
        """

        # if no sim_date is specified, the prediction starts from the last entry of transaction data
        df = self.get_data().copy()
        if sim_date is None:
            idate = pd.to_datetime(df.tail(1)['date'].values).date[0]
        else:
            idate = sim_date
            df.drop(labels=df[df['date'] > str(sim_date)].index, inplace=True)

        # prepare start date for date range average
        sdate = idate - datetime.timedelta(days=90)
        avg = accounts.get_data_range_daily_average(start_date=sdate, end_date=idate)
        avg['balance'] = avg['balance'].apply(lambda i: i*30)

        if self.planned_transactions is not None:
            template = df.iloc[0, :].copy(deep=True)
            bal = self.get_current_balance()

            # for each day in the projection
            for d in range(1, days):
                date = (idate + timedelta(days=d))

                # add planned transaction tuple
                for ix, ptransaction in self.planned_transactions.iterrows():
                    if _is_planned_transaction(year=ptransaction['year'],
                                               month=ptransaction['month'],
                                               day=ptransaction['day'],
                                               date=date):
                        template.loc['date'] = pd.to_datetime(date)
                        template['transaction_num'] = 'na'
                        template['description'] = ptransaction['description']
                        template['interests'] = 0
                        template['advance'] = ptransaction['withdrawal']
                        template['reimboursment'] = ptransaction['deposit']
                        template['balance'] = bal + ptransaction['withdrawal'] - ptransaction['deposit']
                        template['code'] = ptransaction['code']

                        df = df.append(other=template, ignore_index=True)
                        df.sort_values(by=['date', 'transaction_num'], inplace=True)

                        bal = template['balance']

                # add average expenses to prediction
                if date.day == 1:
                    for expense_code in avg.index:
                        if expense_code not in ['internet', 'rent', 'pay', 'cell', 'hydro', 'interest', 'other']:
                            template.loc['date'] = pd.to_datetime(date)
                            template['transaction_num'] = 'na'
                            template['description'] = expense_code
                            template['interests'] = 0
                            template['advance'] = avg.loc[expense_code, 'balance'] * (avg.loc[expense_code, 'balance'] < 0)
                            template['reimboursment'] = avg.loc[expense_code, 'balance'] * (avg.loc[expense_code, 'balance'] >= 0)
                            template['balance'] = bal + template['reimboursment'] - template['advance']
                            template['code'] = expense_code

                            df = df.append(other=template, ignore_index=True)
                            df.sort_values(by=['date', 'transaction_num'], inplace=True)

                            bal = template['balance']

            df['delta'] = df.date.diff().shift(-1)
            df['delta'] = df.delta.array.days
            df['cap_interest'] = np.multiply(df.balance, df.delta) * self.metadata.interest_rate / 372
            df.replace(np.nan, 0, inplace=True)

            interest_sum = 0
            for ix, row in df[self._get_last_interest_payment_index():].iterrows():
                interest_sum += row['cap_interest']
                if row['date'].day == 1 and row['interests'] == 0 and row['code'] == 'interest':
                    df.loc[ix, 'interests'] = interest_sum
                    interest_sum = 0
                elif row['date'].day == 1 \
                        and row['description'] == 'Avance au compte EOP (paiement intérêt)':
                    df.loc[ix, 'advance'] = df.loc[ix - 1, 'interests']
                    df.loc[ix, 'balance'] += df.loc[ix, 'advance']

        else:
            df = None
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

    def plot_prediction(self,
                        start_date: datetime.date = None,
                        sim_date: datetime.date = None,
                        days=90,
                        show=False,
                        fig_size=(7, 8)):

        d = self.get_predicted_balance(days=days, sim_date=sim_date)
        d.loc[:, 'balance'] = -1 * d.loc[:, 'balance'].copy()
        if start_date is None:
            start_date = datetime.datetime.today().date() - timedelta(days=7)
        d = d[d['date'].array.date > start_date]

        edate = d.tail(1)['date'].array[0]

        for ix, row in d.iterrows():
            i = 1
            date = row['date'] + datetime.timedelta(days=i)
            while (d['date'] == date).sum() == 0 and date < edate:
                template = row.copy()
                template['date'] = date
                template['transaction_num'] = row['transaction_num']
                template['description'] = 0
                template['interests'] = 0
                template['advance'] = 0
                template['reimboursment'] = 0
                template['code'] = 0
                template['delta'] = 0
                template['cap_interest'] = 0
                d = d.append(template)
                i += 1
                date = row['date'] + datetime.timedelta(days=i)

        d.sort_values(by='date', inplace=True)

        fig = plt.figure(figsize=fig_size)

        tmp = d.loc[d[d['transaction_num'] == 'na'].index[0]-1:d[d['transaction_num'] == 'na'].index[0]].copy()

        plt.plot(d.loc[d['transaction_num'] != 'na', 'date'],
                 d.loc[d['transaction_num'] != 'na', 'balance'],
                 c='g')
        plt.plot(tmp['date'],
                 tmp['balance'],
                 c='b')
        plt.plot(d.loc[d['transaction_num'] == 'na', 'date'],
                 d.loc[d['transaction_num'] == 'na', 'balance'],
                 c='b')

        plt.title(f'Prediction')

        plt.xticks(rotation=90)
        if show:
            plt.show()
        return fig

    #TODO: Fix this function
    def plot_prediction_compare(self,
                                start_date: datetime.date = None,
                                sim_date: datetime.date = None,
                                days=90,
                                show=False,
                                fig_size=(7, 8)):

        d1 = self.get_predicted_balance(days=days, sim_date=sim_date).copy()
        d1.loc[:, 'balance'] = -1 * d1.loc[:, 'balance']
        if start_date is None:
            start_date = datetime.datetime.today().date() - timedelta(days=7)
        d1 = d1[d1['date'].array.date > start_date]
        d1.loc[:, 'date'] = d1.loc[:, 'date'].apply(lambda i: str(i).replace(' 00:00:00', ''))

        fig = plt.figure(figsize=fig_size)

        tmp = d1.loc[d1[d1['transaction_num']=='na'].index[0]-1:d1[d1['transaction_num']=='na'].index[0]].copy()

        plt.plot(d1.loc[d1['transaction_num'] != 'na', 'date'],
                 d1.loc[d1['transaction_num'] != 'na', 'balance'],
                 c='g')
        plt.plot(tmp['date'],
                 tmp['balance'],
                 c='b')
        plt.plot(d1.loc[d1['transaction_num'] == 'na', 'date'],
                 d1.loc[d1['transaction_num'] == 'na', 'balance'],
                 c='b')

        d2 = self.get_predicted_balance(days=days, sim_date=None).copy()
        d2.loc[:, 'balance'] = -1 * d2.loc[:, 'balance']
        if start_date is None:
            start_date = datetime.datetime.today().date() - timedelta(days=7)
        d2 = d2[d2['date'].array.date > start_date]
        d2.loc[:, 'date'] = d2.loc[:, 'date'].apply(lambda i: str(i).replace(' 00:00:00', ''))

        tmp = d2.loc[d2[d2['transaction_num'] == 'na'].index[0]-1:d2[d2['transaction_num'] == 'na'].index[0]].copy()

        plt.plot(d2.loc[d2['transaction_num'] != 'na', 'date'],
                 d2.loc[d2['transaction_num'] != 'na', 'balance'],
                 c='y')
        plt.plot(tmp['date'],
                 tmp['balance'],
                 c='k')
        plt.plot(d2.loc[d2['transaction_num'] == 'na', 'date'],
                 d2.loc[d2['transaction_num'] == 'na', 'balance'],
                 c='r')

        plt.title(f'Prediction')

        plt.xticks(rotation=90)
        if show:
            plt.show()
        return fig

    def _get_last_interest_payment_date(self) -> datetime.date:
        return (self.transaction_data[self.transaction_data['interests'] > 0]).tail(n=1).date.array.date[0]

    def _get_last_interest_payment_index(self) -> datetime.date:
        return (self.transaction_data[self.transaction_data['interests'] > 0]).tail(n=1).index[0]

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
            d['total'] = np.subtract(d.loc[:, ['reimboursment']], d.loc[:, ['capital_paid']])
            if 'internal_cashflow' in d.index:
                d.drop(labels='internal_cashflow', inplace=True)
            d.drop(columns=['capital_paid', 'reimboursment'], inplace=True)

        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_date_range_daily_average(start_date=start_date, end_date=end_date)
        if d is not None:
            if 'internal_cashflow' in d.index:
                d.drop(labels=['internal_cashflow'], inplace=True)
            d['balance'] = np.subtract(d['reimboursment'], d['capital_paid'])
            d.drop(columns=['transaction_num', 'reimboursment', 'capital_paid'], inplace=True)
        return d

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
                            row['code'] = _get_codes(pd.DataFrame(row).T).values[0]
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
                # TODO make this snippet more elegant
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
        cash_flow['code'] = _get_codes(cash_flow)

        return cash_flow

    def get_summed_data(self, year=None, month=None, day=None):
        d = self.get_data(year=year, month=month, day=day)
        if d.shape[0] > 0:
            d = d.loc[:, ['credit/payment', 'code']].groupby('code').sum()
            if 'internal_cashflow' in d.index:
                d.drop(labels='internal_cashflow', inplace=True)
            d.columns = ['total']
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_date_range_daily_average(start_date=start_date, end_date=end_date)
        if d is not None:
            d.drop(labels=['internal_cashflow'], inplace=True)
            d['balance'] = d['credit/payment']
            d.drop(columns=['credit/payment'], inplace=True)
        return d

    def get_current_balance(self) -> float:
        return self.transaction_data.tail(n=1)['balance'].values[0]

    # TODO implement update method for VISA pp credit card
    def update_from_raw_files(self):
        pass

    # TODO implement prediction method
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

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_date_range_daily_average(start_date=start_date, end_date=end_date)
        if d is not None:
            d.drop(labels=['internal_cashflow'], inplace=True)
            d['balance'] = np.subtract(d['credit'], d['debit'])
            d.drop(columns=['credit', 'debit'], inplace=True)
        return d

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
        cash_flow.drop_duplicates(subset=cash_flow.columns[cash_flow.columns != 'code'], inplace=True)
        cash_flow.sort_values(by=['date'], inplace=True)
        cash_flow.reset_index(inplace=True, drop=True)
        # tst = _get_codes(cash_flow[cash_flow['code'] == 0])
        # cash_flow.iloc[cash_flow['code'] == 0, 'code'] = tst
        for ix, row in cash_flow[cash_flow['code'] == 0].iterrows():
            cash_flow.loc[ix, 'code'] = _get_codes(pd.DataFrame(row).T).values[0]

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

    # TODO refactor to load data in parent class and process it
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

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_date_range_daily_average(start_date=start_date, end_date=end_date)
        if d is not None:
            if 'internal_cashflow' in d.index:
                d.drop(labels=['internal_cashflow'], inplace=True)
            if 'amount' in d.columns:
                d.drop(columns=['amount'], inplace=True)
        return d

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
        cash_flow['code'] = _get_codes(cash_flow)
        l1 = self.transaction_data.shape[0]
        self.transaction_data = self.transaction_data.append(cash_flow)
        self.transaction_data.drop_duplicates(inplace=True)
        self.transaction_data.reset_index(inplace=True, drop=True)
        l2 = self.transaction_data.shape[0]

        if l2 - l1 > 0:
            print(f'{l2 - l1} new entries added')
            if self._save_prompt():
                self.to_pickle()
            self.to_pickle()

    # TODO implement prediction method
    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        df = self.get_data()
        return df


class Accounts:
    def __init__(self, l_accounts_list: list):
        self.accounts_list = l_accounts_list

    def __iter__(self):
        yield from self.accounts_list

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

    def get_daily_average(self, year=None, month=None, day=None):
        av_l = list()
        d = None
        for acc in self.accounts_list:
            t = acc.get_daily_average(year=year, month=month, day=day)
            if t.shape[0] > 0:
                av_l.append(t)
        if len(av_l) > 0:
            d = pd.concat(av_l)
            d = d.groupby(by='code').sum()
        return d

    def get_data_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        av_l = list()
        d = None
        for acc in self.accounts_list:
            t = acc.get_date_range_daily_average(start_date=start_date, end_date=end_date)
            if t is not None:
                av_l.append(t)
        if len(av_l) > 0:
            d = pd.concat(av_l)
            d = d.groupby(by='code').sum()

        if d is not None:
            if 'transaction_num' in d.columns:
                d.drop(columns=['transaction_num'], inplace=True)
            if 'internal_cashflow' in d.index:
                d.drop(labels=['internal_cashflow'], inplace=True)

        return d

    def barplot(self, year=None, month=None, day=None, show=False, average: bool = False):

        if not average:
            d = self.get_summed_data(year=year, month=month, day=day)
        else:
            d = self.get_summed_average(year=year, month=month, day=day)

        if d is not None and d.shape[0] > 0:
            title = f'{year} ' * (year is not None) + \
                    f'{month} ' * (month is not None) + \
                    f'{day} ' * (day is not None) + f'All accounts'
            return _barplot_dataframe(d=d, title=title, show=show)

    def get_names(self):
        name_list = list()
        for acc in self.accounts_list:
            name_list.append(acc.metadata.name.name)
        return name_list

    def get_by_name(self, name: AccountNames):
        res = None
        for acc in self.accounts_list:
            if acc.metadata.name.name == name.name:
                res = acc
                break
        return res


"""
Create account objects
"""
base_dir = os.path.dirname(os.path.abspath(__file__))
os.path.abspath(__file__)
data_dir = os.path.join(base_dir, 'data')
pickle_dir = os.path.join(base_dir, 'pickle_objects')

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
                           # unique_keys=['date', 'transaction_num'],
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
                           interest_rate=0.0295,
                           name=AccountNames.DESJARDINS_MC,
                           columns_names=names,
                           # unique_keys=['date', 'transaction_num'],
                           type=AccountType.CREDIT,
                           status=AccountStatus.OPEN)
desjardins_mc = DesjardinsMC(lmetadata=metadata)

# ==============
# Desjardins PR
# ==============
metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'desjardins_csv_files'),
                           serialized_object_path=os.path.join(pickle_dir, 'desjardins_pr.pkl'),
                           planned_transactions_path=os.path.join(data_dir,
                                                                  'desjardins_planned_transactions.pkl'),
                           interest_rate=0.045,
                           name=AccountNames.DESJARDINS_PR,
                           columns_names=names,
                           # unique_keys=['date', 'transaction_num'],
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
                                                                  'desjardins_planned_transactions.pkl'),
                           interest_rate=0,
                           name=AccountNames.VISA_PP,
                           columns_names=names,
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
                                                                  'desjardins_planned_transactions.pkl'),
                           interest_rate=0,
                           name=AccountNames.CAPITAL_ONE,
                           columns_names=names,
                           type=AccountType.CREDIT,
                           status=AccountStatus.OPEN)
capital_one = CapitalOne(lmetadata=metadata)

# ==============
# PayPal
# ==============
names = ['Date', 'Time', 'TimeZone', 'Name', 'description', 'Status', 'Currency', 'Amount', 'Receipt ID', 'Balance']
metadata = AccountMetadata(raw_files_dir=os.path.join(data_dir, 'paypal_csv_files'),
                           serialized_object_path=os.path.join(pickle_dir, 'paypal.pkl'),
                           planned_transactions_path=os.path.join(data_dir,
                                                                  'desjardins_planned_transactions.pkl'),
                           interest_rate=0,
                           name=AccountNames.PAYPAL,
                           columns_names=names,
                           type=AccountType.PREPAID,
                           status=AccountStatus.OPEN)
paypal = PayPal(lmetadata=metadata)

accounts_list = list()

accounts_list.append(desjardins_op)
accounts_list.append(desjardins_mc)
accounts_list.append(desjardins_pr)
accounts_list.append(visapp)
accounts_list.append(capital_one)
accounts_list.append(paypal)

accounts = Accounts(l_accounts_list=accounts_list.copy())
del accounts_list
