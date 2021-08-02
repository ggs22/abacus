import numpy as np
import pandas as pd
import pickle
import colorama
import os
import datetime
import re

from dataclasses import dataclass
from enum import Enum
from abc import abstractmethod
from datetime import timedelta
from abc import ABC


class AccountNames(Enum):
    DESJARDINSOP = 1
    DESJARDINSMC = 2
    DESJARDINSPR = 3
    VISAPP = 4
    PAYPAL = 5
    CAPITALONE = 6


class AccountStatus(Enum):
    OPEN = 1
    CLOSED = 2


class AccountType(Enum):
    OPERATIONS = 1
    CREDIT = 2
    PREPAID = 3


@dataclass
class AccountMetadata:
    raw_files_path: str
    serialized_object_path: str
    planned_transactions_path: str
    interest_rate: float
    name: AccountNames
    columns_names: list
    type: AccountType
    status: AccountStatus


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

    assignations = pd.read_csv('data/assignations.csv', encoding='utf-8', sep=',').dropna(axis=1, how='all')
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
    def __init__(self, metadata: AccountMetadata):
        df = None
        self.metadata = metadata
        if os.path.exists(self.metadata.serialized_object_path):
            with open(self.metadata.serialized_object_path, 'rb') as save_file:
                df = pickle.load(save_file)

        elif os.path.exists(self.metadata.raw_files_path):
            df = self._load_from_raw_files()

        if df is None:
            raise FileNotFoundError(f'No serialized or raw data (in {self.metadata.raw_files_path}) was found for '
                                    f'{self.metadata.name}')
        else:
            self.transaction_data = df
        self.planned_transactions = self._load_planned_transactions()

    def get_data_by_date(self, year=None, month=None, day=None):
        """
        Returns transactions for a specified time period. If year, month and day are
        not specified, defaults to all current year transactions

        :param year: year of the requested data, defaults to current year
        :param month: month of the requested data, defaults to current month
        :param day: day of the requested data, defaults to current day
        :return: Pandas Dataframe containing transactions for a specified time period
        """
        if year is None:
            year = datetime.datetime.today().year
        _res = self.transaction_data[self.transaction_data['date'].array.year == year]

        if month is not None:
            _res = _res[_res['date'].array.month == month]

        # if day is none we get transaction for the whole month
        if day is not None:
            _res = _res[_res['date'].array.day == day]

        return _res

    def _load_from_raw_files(self) -> pd.DataFrame:
        """
        Laods the data from CSV files and returns a dataframe with the
        corresponding data

        :param names: CSV column names
        :param input_path: The path of the csv file containing transactions data
        :return: A Pandas DataFrame containing the transactions data
        """

        # TODO keep a record of processed raw files as to avoid reprocessing them in updates

        if self.metadata.raw_files_path == '':
            raise ValueError('please specify a path for Desjardins CSV files')

        csv_files = os.listdir(self.metadata.raw_files_path)
        cash_flow = pd.DataFrame(columns=self.metadata.columns_names)
        for x in csv_files:
            if x[-4:] == '.csv':
                cash_flow = cash_flow.append(pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_path, x),
                                                         encoding='latin1',
                                                         names=self.metadata.columns_names),
                                             ignore_index=True)

        # Convert strings to actual date time objects
        if 'date' in cash_flow.columns:
            cash_flow['date'] = pd.to_datetime(cash_flow['date'], format='%Y-%m-%d')

        # Adds column,and inputs transaction code
        cash_flow['code'] = _get_codes(cash_flow)
        cash_flow = cash_flow.replace(np.nan, 0)

        return cash_flow

    def _load_planned_transactions(self) -> pd.DataFrame:
        df = None
        if os.path.exists(self.metadata.planned_transactions_path):
            df = pd.read_csv(self.metadata.planned_transactions_path)
        return df

    @staticmethod
    def _save_prompt():
        ans = None
        while ans not in ['y', 'n', '']:
            ans = input('Save new entries?([y]/n)').lower()
        return ans in ['y', '']

    def to_pickle(self):
        with open(self.metadata.serialized_object_path, 'wb') as save_file:
            pickle.dump(self.transaction_data, save_file)

    def get_data(self) -> pd.DataFrame:
        if self.col_mask is not None:
            return self.transaction_data[self.col_mask]
        else:
            return self.transaction_data[self.metadata.columns_names]

    @abstractmethod
    def get_current_balance(self):
        """Each account type has its own balance calculation"""

    @abstractmethod
    def update_from_raw_files(self):
        """Each account type has its own update method"""

    @abstractmethod
    def get_predicted_balance(self):
        """Each account type has its own prediction method"""


class DesjardinsOP(Account):
    def __init__(self, metadata: AccountMetadata):
        super().__init__(metadata=metadata)
        self.col_mask = ['date', 'transaction_num', 'description', 'withdrawal', 'deposit', 'balance', 'code']
        self.transaction_data = self.transaction_data[self.transaction_data['account'] == 'EOP']

    def get_current_balance(self) -> float:
        return self.transaction_data.tail(n=1)['balance'].values[0]

    def update_from_raw_files(self):
        new_entries = 0
        csv_files = os.listdir(self.metadata.raw_files_path)
        for x in csv_files:
            if x[-4:] == '.csv':
                new = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_path, x),
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


class DesjardinsMC(Account):
    def __init__(self, metadata: AccountMetadata):
        super().__init__(metadata=metadata)
        self.col_mask = ['date', 'transaction_num', 'description', 'interests', 'advance', 'reimboursment', 'balance',
                         'code']
        self.transaction_data = self.transaction_data[self.transaction_data['account'] == 'MC2']

    def get_current_balance(self) -> float:
        return self.transaction_data.tail(n=1)['balance'].values[0]

    def update_from_raw_files(self):
        new_entries = 0
        csv_files = os.listdir(self.metadata.raw_files_path)
        # cash_flow = pd.DataFrame()
        for x in csv_files:
            if x[-4:] == '.csv':
                new = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_path, x),
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
                            self.transaction_data.append(other=row, ignore_index=True)
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
                        template['interests'] = self._estimate_interests()
                        template['advance'] = ptransaction['withdrawal']
                        template['reimboursment'] = ptransaction['deposit']
                        template['balance'] = bal + template['interests'] + ptransaction['withdrawal'] - ptransaction['deposit']
                        template['code'] = 'planned'
                        df = df.append(other=template, ignore_index=True)
                        df.sort_values(by=['date', 'transaction_num'], inplace=True)

                        bal = template['balance']
        else:
            df = None
        return df

    def _estimate_interests(self):
        # TODO implement interest estimation method
        return 0


class DesjardinsPR(Account):
    def __init__(self, metadata: AccountMetadata):
        super().__init__(metadata=metadata)
        self.col_mask = ['date', 'transaction_num', 'description', 'capital_paid', 'reimboursment', 'balance', 'code']
        self.transaction_data = self.transaction_data[self.transaction_data['account'] == 'PR1']

    def get_current_balance(self) -> float:
        return self.transaction_data.tail(n=1)['balance'].values[0]

    def update_from_raw_files(self):
        new_entries = 0
        csv_files = os.listdir(self.metadata.raw_files_path)
        for x in csv_files:
            if x[-4:] == '.csv':
                new = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_path, x),
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
                            self.transaction_data.append(other=row, ignore_index=True)
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

    def _estimate_interests(self):
        # TODO implement interest estimation method
        return 0


class VisaPP(Account):
    def __init__(self, metadata: AccountMetadata):
        super().__init__(metadata=metadata)
        self.col_mask = ['date', 'transaction_num', 'description', 'credit/payment', 'balance', 'code']

    def _load_from_raw_files(self) -> pd.DataFrame:
        """
            This function returns an aggloromated dataframe from dejardins prepard credit card pdf files
            args:
                - input_path        location of pdf files to be agglomerated
            """

        # pdf_files = Path(input_path).glob('*.pdf')
        pdf_files = os.listdir(self.metadata.raw_files_path)

        # Desjardins ppcard pdf files columns names
        names = ['date', 'transaction_num', 'description', 'credit/payment']
        cash_flow = pd.DataFrame(columns=names)

        suffix1 = '_processed.txt'
        for x in pdf_files:
            if x[-len(suffix1):] == suffix1:
                tot_df = pd.read_csv(os.path.join(self.metadata.raw_files_path, x),
                                     sep=';',
                                     encoding='utf-8',
                                     names=names,
                                     header=None)
                start_index = tot_df[tot_df['date'] == 'Jour'].index[0] + 1
                mid_index = tot_df[tot_df['date'] == 'Total'].index[0]
                end_index = tot_df[tot_df['date'] == 'SOLDE PRÉCÉDENT'].index[0]
                year = int(tot_df.iloc[1, 1])

                expenses = tot_df[start_index:mid_index].copy()
                payments = tot_df[mid_index + 1:end_index].copy()

                expenses.loc[:, 'transaction_num'] = expenses.loc[:, 'transaction_num'].apply(lambda i: str(int(i)) + 'e')
                payments.loc[:, 'transaction_num'] = payments.loc[:, 'transaction_num'].apply(lambda i: str(int(i)) + 'p')

                cash_flow = cash_flow.append(expenses, ignore_index=True)
                cash_flow = cash_flow.append(payments, ignore_index=True)

            elif x[-4:] == '.txt':
                # TODO make this snippet more elegant
                tnumber_prefixes, ix = ['e', 'p'], 0
                with open(os.path.join(self.metadata.raw_files_path, x), 'r') as raw_file:
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
                                                                             index=names),
                                                             ignore_index=True)
                            ix += 1

        # Adds column,and inputs transaction code
        cash_flow = cash_flow.replace(np.nan, 0)
        cash_flow['date'] = pd.to_datetime(cash_flow['date'])
        cash_flow.sort_values(by=['date', 'transaction_num'], ascending=True, inplace=True)
        cash_flow['credit/payment'] = pd.to_numeric(cash_flow['credit/payment'])
        cash_flow['balance'] = cash_flow['credit/payment'].cumsum()
        cash_flow['code'] = _get_codes(cash_flow)

        return cash_flow

    def get_current_balance(self) -> float:
        return self.transaction_data.tail(n=1)['balance'].values[0]

    def update_from_raw_files(self):
        new_entries = 0
        # TODO implement update method for VISA prepaid account (can be implemented in 'laod_from_raw_files')
        # csv_files = os.listdir(self.metadata.raw_files_path)
        # for x in csv_files:
        #     if x[-4:] == '.csv':
        #         new = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_path, x),
        #                           encoding='latin1',
        #                           names=self.metadata.columns_names)
        #         for ix, row in new.iterrows():
        #
        #             if (row['date'], row['transaction_num']) in self.transaction_data.set_index(
        #                     keys=['date', 'transaction_num']).index:
        #                 continue
        #             else:
        #                 if row['account'] == 'PR1':
        #                     row['date'] = pd.to_datetime(row['date'], format='%Y-%m-%d')
        #                     row['code'] = _get_codes(pd.DataFrame(row).T).values[0]
        #                     row.replace(np.nan, 0, inplace=True)
        #                     self.transaction_data.append(other=row, ignore_index=True)
        #                     new_entries += 1
        #
        # if new_entries > 0:
        #     self.transaction_data.drop_duplicates(keep='first', subset=self.metadata.columns_names, inplace=True)
        #     self.transaction_data.sort_values(by=['date', 'transaction_num'], inplace=True)
        #
        #     print(f'{new_entries} new entries added')
        #     if self._save_prompt():
        #         self.to_pickle()

    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        df = self.get_data()
        # TODO implement prediction method
        # if self.planned_transactions is not None:
        #     template = df.iloc[0, :].copy(deep=True)
        #     bal = self.get_current_balance()
        #     for d in range(0, days):
        #         date = (datetime.datetime.today().date() + timedelta(days=d))
        #         for ix, ptransaction in self.planned_transactions.iterrows():
        #             if _is_planned_transaction(year=ptransaction['year'],
        #                                        month=ptransaction['month'],
        #                                        day=ptransaction['day'],
        #                                        date=date):
        #                 template.loc['date'] = date
        #                 template['transaction_num'] = 'na'
        #                 template['description'] = ptransaction['description']
        #                 template['paid_capital'] = ptransaction['withdrawal']
        #                 template['reimboursment'] = ptransaction['deposit']
        #                 template['balance'] = bal + ptransaction['paid_capital'] - ptransaction['reimboursment']
        #                 template['code'] = 'planned'
        #                 df = df.append(other=template, ignore_index=True)
        #                 df.sort_values(by=['date', 'transaction_num'], inplace=True)
        #
        #                 bal = template['balance']
        # else:
        #     df = None
        return df


class CapitalOne(Account):
    def __init__(self, metadata: AccountMetadata):
        super().__init__(metadata=metadata)
        self.col_mask = ['date', 'transaction_num', 'description', 'credit/payment', 'balance', 'code']

    def _load_from_raw_files(self) -> pd.DataFrame:
        """
            This function returns an aggloromated dataframe from dejardins prepard credit card pdf files
            args:
                - input_path        location of pdf files to be agglomerated
            """

        # pdf_files = Path(input_path).glob('*.pdf')
        pdf_files = os.listdir(self.metadata.raw_files_path)

        # Desjardins ppcard pdf files columns names
        names = ['date', 'transaction_num', 'description', 'credit/payment']
        cash_flow = pd.DataFrame(columns=names)

        suffix1 = '_processed.txt'
        for x in pdf_files:
            if x[-len(suffix1):] == suffix1:
                tot_df = pd.read_csv(os.path.join(self.metadata.raw_files_path, x),
                                     sep=';',
                                     encoding='utf-8',
                                     names=names,
                                     header=None)
                start_index = tot_df[tot_df['date'] == 'Jour'].index[0] + 1
                mid_index = tot_df[tot_df['date'] == 'Total'].index[0]
                end_index = tot_df[tot_df['date'] == 'SOLDE PRÉCÉDENT'].index[0]
                year = int(tot_df.iloc[1, 1])

                expenses = tot_df[start_index:mid_index].copy()
                payments = tot_df[mid_index + 1:end_index].copy()

                expenses.loc[:, 'transaction_num'] = expenses.loc[:, 'transaction_num'].apply(lambda i: str(int(i)) + 'e')
                payments.loc[:, 'transaction_num'] = payments.loc[:, 'transaction_num'].apply(lambda i: str(int(i)) + 'p')

                cash_flow = cash_flow.append(expenses, ignore_index=True)
                cash_flow = cash_flow.append(payments, ignore_index=True)

            elif x[-4:] == '.txt':
                # TODO make this snippet more elegant
                tnumber_prefixes, ix = ['e', 'p'], 0
                with open(os.path.join(self.metadata.raw_files_path, x), 'r') as raw_file:
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
                                                                             index=names),
                                                             ignore_index=True)
                            ix += 1

        # Adds column,and inputs transaction code
        cash_flow = cash_flow.replace(np.nan, 0)
        cash_flow['date'] = pd.to_datetime(cash_flow['date'])
        cash_flow.sort_values(by=['date', 'transaction_num'], ascending=True, inplace=True)
        cash_flow['credit/payment'] = pd.to_numeric(cash_flow['credit/payment'])
        cash_flow['balance'] = cash_flow['credit/payment'].cumsum()
        cash_flow['code'] = _get_codes(cash_flow)

        return cash_flow

    def get_current_balance(self) -> float:
        return self.transaction_data.tail(n=1)['balance'].values[0]

    def update_from_raw_files(self):
        new_entries = 0
        # TODO implement update method for VISA prepaid account (can be implemented in 'laod_from_raw_files')
        # csv_files = os.listdir(self.metadata.raw_files_path)
        # for x in csv_files:
        #     if x[-4:] == '.csv':
        #         new = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_path, x),
        #                           encoding='latin1',
        #                           names=self.metadata.columns_names)
        #         for ix, row in new.iterrows():
        #
        #             if (row['date'], row['transaction_num']) in self.transaction_data.set_index(
        #                     keys=['date', 'transaction_num']).index:
        #                 continue
        #             else:
        #                 if row['account'] == 'PR1':
        #                     row['date'] = pd.to_datetime(row['date'], format='%Y-%m-%d')
        #                     row['code'] = _get_codes(pd.DataFrame(row).T).values[0]
        #                     row.replace(np.nan, 0, inplace=True)
        #                     self.transaction_data.append(other=row, ignore_index=True)
        #                     new_entries += 1
        #
        # if new_entries > 0:
        #     self.transaction_data.drop_duplicates(keep='first', subset=self.metadata.columns_names, inplace=True)
        #     self.transaction_data.sort_values(by=['date', 'transaction_num'], inplace=True)
        #
        #     print(f'{new_entries} new entries added')
        #     if self._save_prompt():
        #         self.to_pickle()

    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        df = self.get_data()
        # TODO implement prediction method
        # if self.planned_transactions is not None:
        #     template = df.iloc[0, :].copy(deep=True)
        #     bal = self.get_current_balance()
        #     for d in range(0, days):
        #         date = (datetime.datetime.today().date() + timedelta(days=d))
        #         for ix, ptransaction in self.planned_transactions.iterrows():
        #             if _is_planned_transaction(year=ptransaction['year'],
        #                                        month=ptransaction['month'],
        #                                        day=ptransaction['day'],
        #                                        date=date):
        #                 template.loc['date'] = date
        #                 template['transaction_num'] = 'na'
        #                 template['description'] = ptransaction['description']
        #                 template['paid_capital'] = ptransaction['withdrawal']
        #                 template['reimboursment'] = ptransaction['deposit']
        #                 template['balance'] = bal + ptransaction['paid_capital'] - ptransaction['reimboursment']
        #                 template['code'] = 'planned'
        #                 df = df.append(other=template, ignore_index=True)
        #                 df.sort_values(by=['date', 'transaction_num'], inplace=True)
        #
        #                 bal = template['balance']
        # else:
        #     df = None
        return df
