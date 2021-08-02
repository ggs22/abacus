import numpy as np
import pandas as pd
import pickle
import colorama
import re
import os
import datetime

# from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from abc import abstractmethod
from datetime import timedelta
from abc import ABC
from typing import Optional


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


def _load_desjardins_ppcard_pdf_files(input_path) -> pd.DataFrame:
    """
    This function returns an aggloromated dataframe from dejardins prepard credit card pdf files
    args:
        - input_path        location of pdf files to be agglomerated
    """

    pdf_files = Path(input_path).glob('*.pdf')

    # 1#2#3#4#5## <object_str> -#.##
    withdraw_pattern = '([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{3})([a-zA-Z].{,49})(-[0-9]+,[0-9]{2})'
    # 1#2#3#4#5##PAIMENT CAISSE #.##
    payment_pattern = '([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{3})(PAIEMENT CAISSE *)([0-9]+,[0-9]{2})'
    # 20##
    year_pattern = '^(20[0-9]{2})'

    # Desjardins ppcard pdf files columns names
    names = ['date', 'transaction_num', 'description', 'credit/payment']
    _cash_flow = pd.DataFrame(columns=names)

    if input_path == '':
        raise ValueError('please specify a path for Desjardins prepaid credit card pdf files')

    for x in pdf_files:
        if x.is_file():
            reader = PyPDF2.PdfFileReader(stream=open(x, 'rb'))
            # since pypdf2 doest not a prefect text extraction, we need to get year from document info
            regx_year = re.compile(year_pattern)
            year = int(regx_year.findall(reader.documentInfo['/CreationDate'])[0])

            for page_num in range(0, reader.getNumPages()):
                page = reader.getPage(page_num).extractText()
                regx_withdrawal = re.compile(withdraw_pattern)
                regx_payment = re.compile(payment_pattern)
                for r_w in [*regx_withdrawal.findall(page), *regx_payment.findall(page)]:
                    _serie = pd.Series(index=names)
                    # text to date
                    day = int(r_w[0])
                    month = int(r_w[1])
                    if month == 12:
                        year -= 1
                    _serie['date'] = datetime.date(year=year, month=month, day=day)
                    _serie['transaction_num'] = int(r_w[4])
                    _serie['description'] = r_w[5]
                    _serie['credit/payment'] = float(r_w[6].replace(',', '.'))
                    _cash_flow = _cash_flow.append(_serie, ignore_index=True)

    # Adds column,and inputs transaction code
    _cash_flow['code'] = _get_codes(_cash_flow)
    _cash_flow = _cash_flow.replace(np.nan, 0)

    return _cash_flow


def _load_capital_one_csv_files(input_path) -> pd.DataFrame:
    """
    This function returns an aggloromated dataframe from capital one csv files

    args:
        - input_path        location of csv files to be agglomerated
    """

    if input_path == '':
        raise ValueError('please specify a path for Capital One CSV files')

    # Capital One csv files columns names
    names = ['date', 'posted_date', 'card_num', 'description', 'category', 'debit', 'credit']

    csv_files = Path(input_path).glob('*.csv')
    _cash_flow = pd.DataFrame(columns=names)
    for x in csv_files:
        if x.is_file():
            _cash_flow = _cash_flow.append(pd.read_csv(x.as_posix(), encoding='latin1', names=names, header=0),
                                           ignore_index=True)

    # Convert strings to actual date time objects
    _cash_flow['date'] = pd.to_datetime(_cash_flow['date'])
    _cash_flow['code'] = _get_codes(_cash_flow)
    _cash_flow = _cash_flow.replace(np.nan, 0)

    del _cash_flow['posted_date']

    return _cash_flow


class CSVParser:
    """
    This classes is responsible for acquiring bank accounts transactions from csv files and parsing it into manageable
    pandas Dataframes.
        Theses accounts are suppported:
            - Desjardins OP account (Operations)
            - Desjardins CL account (Credit Line)
            - Desjardins SL account (Student Loan)
            - Costco Capital One
    """

    def __init__(self, desjardins_input_path="", desjardins_ppcard_input_path="", capital_one_input_path=""):

        self.accounts = dict()

        self._raw_desjardins_data = None
        self._raw_capital_one_data = None
        self._raw_desjardins_ppcard_data = None

        self._pickle_objects_root = 'pickle_objects/'
        self._desjardins_input_path = desjardins_input_path
        self._desjardins_ppcard_input_path = desjardins_ppcard_input_path
        self._capital_one_input_path = capital_one_input_path

        self._load_accounts_data()
        self._split_accounts()

        _desjardins_file_name = self._pickle_objects_root + "desjardins.pkl"
        if not Path(_desjardins_file_name).exists():
            self.save_desjarding_data()

        _desjardins_ppcard_file_name = self._pickle_objects_root + "desjardins_ppcard.pkl"
        if not Path(_desjardins_ppcard_file_name).exists():
            self.save_desjarding_ppcard_data()

        _capital_one_file_name = self._pickle_objects_root + "capital_one.pkl"
        if not Path(_capital_one_file_name).exists():
            self.save_capital_one_data()

    def _load_accounts_data(self):
        self._load_desjardins_data()
        self._load_desjardins_ppcard_data()
        self._load_capital_one_data()

    def _split_accounts(self):
        """This function parse the raw data stored in pandas dataframes on loading. It stores it in "Account" objects
        which inherits from Dataframes and come with some relevant add-on functions. The account objects are then place
        a dictionnary.
        """
        tmp_df = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'EOP', :]
        # self._operations_account_data = Account(self._operations_account_data.dropna(axis=1, how='all'))
        self.accounts['desjardins_op'] = Account(tmp_df.dropna(axis=1, how='all'))

        tmp_df = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'MC2', :]
        # self._credit_line_data = Account(self._credit_line_data.dropna(axis=1, how='all'))
        self.accounts['desjardins_mc'] = Account(tmp_df.dropna(axis=1, how='all'))

        tmp_df = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'PR1', :]
        # self._student_loan_data = Account(self._student_loan_data.dropna(axis=1, how='all'))
        self.accounts['desjardins_sl'] = Account(tmp_df.dropna(axis=1, how='all'))

        tmp_df = Account(self._raw_desjardins_ppcard_data)
        self.accounts['desjardins_pp'] = Account(tmp_df)

        tmp_df = Account(self._raw_capital_one_data.dropna(axis=1, how='all', inplace=False))
        self.accounts['capital_one'] = Account(tmp_df.dropna(axis=1, how='all', inplace=False))

    def get_accounts(self):
        return self.accounts

    def _load_desjardins_ppcard_data(self):
        """Load Desjardins preparid credit cardt data from pickle object or pdf file"""
        _desjardins_file_name = self._pickle_objects_root + "desjardins_ppcard.pkl"

        if Path(_desjardins_file_name).exists():
            with open(_desjardins_file_name, 'rb') as _df_file:
                self._raw_desjardins_ppcard_data = pickle.load(_df_file)
        else:
            if self._desjardins_ppcard_input_path != "":
                self._raw_desjardins_ppcard_data = _load_desjardins_ppcard_pdf_files(self._desjardins_ppcard_input_path)

        self._raw_desjardins_ppcard_data = self._raw_desjardins_ppcard_data.sort_values(by=['date', 'transaction_num'])

    def _load_desjardins_data(self):
        """Load Desjardins data from pickle object or csv file"""
        _desjardins_file_name = self._pickle_objects_root + "desjardins.pkl"

        if Path(_desjardins_file_name).exists():
            with open(_desjardins_file_name, 'rb') as _df_file:
                self._raw_desjardins_data = Account(pickle.load(_df_file))
        else:
            if self._desjardins_input_path != "":
                self._raw_desjardins_data = _load_desjardins_csv_files(self._desjardins_input_path)

        self._raw_desjardins_data = self._raw_desjardins_data.sort_values(by=['account', 'date', 'transaction_num'])

    def _load_capital_one_data(self):
        """Load Capital One data from pickle object or csv file"""
        _capital_one_file_name = self._pickle_objects_root + "capital_one.pkl"

        if Path(_capital_one_file_name).exists():
            with open(_capital_one_file_name, 'rb') as _df_file:
                self._raw_capital_one_data = pickle.load(_df_file)
        else:
            if self._desjardins_input_path != "":
                self._raw_capital_one_data = _load_capital_one_csv_files(self._capital_one_input_path)

        self._raw_capital_one_data = self._raw_capital_one_data.sort_values(by='date')

    def get_combine_op_and_co(self, year=None, month=None, day=None):
        """
        This function combines the data from the Desjardins operations acount and the Capital One credit account,
        making the necessary column's names changes. It returns the data for all available dates
        :return:    Dataframe containing the combined transaction data from Desjardins OP account and CApital
                    One Credit account
        """
        _data = None
        op, _, _, co = self.get_data_by_date(year=year, month=month, day=day)

        if op.shape[0] > 0:
            _data = self._raw_desjardins_data.copy()
            _data['debit'] = _data['withdrawal']
            _data['credit'] = _data['deposit']
            del _data['withdrawal']
            del _data['deposit']

            _data = _data[_data['code'] != 'internal_cashflow']
        if co.shape[0] > 0:
            if _data is not None:
                _data = pd.concat([_data, self._raw_capital_one_data.copy()], axis=0)
            else:
                _data = self._raw_capital_one_data.copy()

        _data = _data[['date', 'description', 'credit', 'debit', 'code']]
        _data.sort_values(by='date', inplace=True, ignore_index=True)
        _data[['sum credit', 'sum debit']] = _data[['credit', 'debit']].cumsum(axis=0, )
        return _data

    def update_desjardins_data(self, input_path=None):
        """Update Desjardins data from csv file"""

        if input_path is None:
            input_path = self._desjardins_input_path

        if not Path(input_path).exists():
            raise ValueError('please specify a path for Desjardins CSV files')

        # Desjardins csv files columns names
        names = ['branch', 'foliot', 'account', 'date', 'transaction_num', 'description', 'fees', 'withdrawal',
                 'deposit', 'interests', 'capital_paid', 'advance', 'reimboursment', 'balance']

        csv_files = Path(input_path).glob('*.csv')
        _cash_flow = pd.DataFrame()
        for x in csv_files:
            if x.is_file():
                _new = pd.read_csv(x.as_posix(), encoding='latin1', names=names)
                for _ix, row in _new.iterrows():
                    if (row['date'], row['transaction_num']) in self._raw_desjardins_data.set_index(
                            keys=['date', 'transaction_num']).index:
                        continue
                    else:
                        _cash_flow = _cash_flow.append(row, ignore_index=True)

        # Convert strings to actual date time objects
        _cash_flow['date'] = pd.to_datetime.datetime(_cash_flow['date'], format='%Y-%m-%d')
        _cash_flow['code'] = _get_codes(_cash_flow)
        _cash_flow = _cash_flow.replace(np.nan, 0)

        self._raw_desjardins_data = pd.concat([self._raw_desjardins_data, _cash_flow], axis=0)
        self._raw_desjardins_data.drop_duplicates(keep='first', subset=names, inplace=True)
        self._raw_desjardins_data.sort_values(by='date', inplace=True)

        self._split_accounts()
        self.save_desjarding_data()

    def update_desjardins_data_from_clipboard(self):
        """
        Update Desjardins data from clipboard-contained information (when copying data that is not yet on a statement
        from Desjardins)
        """

        # Desjardins files columns names
        names = ['date', 'description', 'withdrawal', 'deposit', 'balance']

        try:
            _cash_flow = pd.read_clipboard(names=names)

            # Convert strings to actual date time objects
            _cash_flow['date'] = pd.to_datetime(_cash_flow['date'])
            _cash_flow = _cash_flow.replace(np.nan, 0)

            self._raw_desjardins_data = pd.concat([self._raw_desjardins_data, _cash_flow], axis=0)
            self._raw_desjardins_data.drop_duplicates(keep='first', subset=names, inplace=True)
            self._raw_desjardins_data.sort_values(by='date', inplace=True)

        except BufferError:
            print(colorama.Fore.YELLOW + "Clipboard content is not recognized..." + colorama.Fore.RESET)

    def update_capital_one_data(self, input_path):
        """Update Capital One data from new csv file(s)"""

        if input_path == '':
            raise ValueError('please specify a path for Capital One CSV files')

        # Capital One csv files columns names
        names = ['date', 'posted_date', 'card_num', 'description', 'category', 'debit', 'credit']

        csv_files = Path(input_path).glob('*.csv')
        _cash_flow = pd.DataFrame(columns=names)
        for x in csv_files:
            if x.is_file():
                _cash_flow = _cash_flow.append(pd.read_csv(x.as_posix(), encoding='latin1', names=names, header=0),
                                               ignore_index=True)

        # Convert strings to actual date time objects
        _cash_flow['date'] = pd.to_datetime(_cash_flow['date'], format='%Y-%m-%d')
        _cash_flow = _cash_flow.replace(np.nan, 0)

        self._raw_capital_one_data = pd.concat([self._raw_capital_one_data, _cash_flow], axis=0)
        self._raw_capital_one_data.drop_duplicates(keep='first', subset=names, inplace=True)
        self._raw_capital_one_data.sort_values(by='date', inplace=True)
        self._raw_capital_one_data.sort_values['posted_date']

    def save_desjarding_data(self):
        """Save Desjardins data as pickle object"""
        Path(self._pickle_objects_root).mkdir(exist_ok=True)
        _desjardins_file_name = self._pickle_objects_root + "desjardins.pkl"
        with open(_desjardins_file_name, 'wb') as _df_file:
            pickle.dump(self._raw_desjardins_data, _df_file, pickle.HIGHEST_PROTOCOL)

    def save_desjarding_ppcard_data(self):
        """Save Desjardins data as pickle object"""
        Path(self._pickle_objects_root).mkdir(exist_ok=True)
        _desjardins_file_name = self._pickle_objects_root + "desjardins_ppcard.pkl"
        with open(_desjardins_file_name, 'wb') as _df_file:
            pickle.dump(self._raw_desjardins_ppcard_data, _df_file, pickle.HIGHEST_PROTOCOL)

    def save_capital_one_data(self):
        """Save Capital One data as pickle object"""
        Path(self._pickle_objects_root).mkdir(exist_ok=True)
        _capital_one_file_name = self._pickle_objects_root + "capital_one.pkl"
        with open(_capital_one_file_name, 'wb') as _df_file:
            pickle.dump(self._raw_capital_one_data, _df_file, pickle.HIGHEST_PROTOCOL)

    def save_all_accounts(self):
        self.save_desjarding_data()
        self.save_desjarding_ppcard_data()
        self.save_capital_one_data()
