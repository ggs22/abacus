import numpy as np
import pandas as pd
import pickle
import datetime
import colorama

from pathlib import Path


def _print_codes_menu(codes, transaction):
    '''
    Prints a CLI menu for manual transaction code assignation
    :param codes: List of possible transaction codes
    :param transaction: Transaction for wich a code assignation is needed
    '''
    print(f'Choose transaction code for:\n'
          f'{colorama.Fore.YELLOW} {transaction} {colorama.Fore.RESET}\n'
          f'(enter corresponding number or "na"):')
    for index, code in enumerate(codes):
        print(f'{index + 1}- {code}', end=(' ' * (25 - len(f'{index + 1}- {code}')) ))
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
    assignations = pd.read_csv('assignations.csv', encoding='utf-8', sep=',').dropna(axis=1, how='all')
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
                    except:
                        print(f'{colorama.Fore.RED}Error: {colorama.Fore.RESET}'
                              f'Please enter a number between 1 and {len(assignations.columns)}')
                else:
                    show_menu = False

    return pd.Series(codes)


def _load_desjardins_csv_files(input_path) -> pd.DataFrame:
    """
    This function returns an aggloromated dataframe from dejardins csv files


    args:
        - input_path        location of csv files to be agglomerated
    """

    if input_path == '':
        raise ValueError('please specify a path for Desjardins CSV files')

    # Desjardins csv files columns names
    names = ['branch', 'foliot', 'account', 'date', 'transaction_num', 'description', 'fees', 'withdrawal',
             'deposit', 'interests', 'capital_paid', 'advance', 'reimboursment', 'balance']

    csv_files = Path(input_path).glob('*.csv')
    _cash_flow = pd.DataFrame(columns=names)
    for x in csv_files:
        if x.is_file():
            _cash_flow = _cash_flow.append(pd.read_csv(x.as_posix(),
                                                       encoding='latin1',
                                                       names=names),
                                           ignore_index=True)

    # Convert strings to actual date time objects
    _cash_flow['date'] = pd.to_datetime(_cash_flow['date'], format='%Y-%m-%d')

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


class CSV_Parser:
    '''
    This classes is responsible for acquiring bank accounts transactions from csv file and parse it into manageable
    pandas Dataframes.
    '''
    def __init__(self, desjardins_input_path="", capital_one_input_path=""):

        self._pickle_objects_root = 'pickle_objects/'
        self._desjardins_input_path = desjardins_input_path
        self._capital_one_input_path = capital_one_input_path

        self._load_desjardins_data()
        self._load_capital_one_data()

        self._operations_account_data = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'EOP', :]
        self._credit_line_data = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'MC2', :]
        self._student_loan_data = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'PR1', :]

        self._operations_account_data = self._operations_account_data.dropna(axis=1, how='all')
        self._credit_line_data = self._credit_line_data.dropna(axis=1, how='all')
        self._student_loan_data = self._student_loan_data.dropna(axis=1, how='all')

        self._capital_one_data = self._raw_capital_one_data.dropna(axis=1, how='all', inplace=False)

        _desjardins_file_name = self._pickle_objects_root + "desjardins.pickle"
        if not Path(_desjardins_file_name).exists():
            self.save_desjarding_data()

        _capital_one_file_name = self._pickle_objects_root + "capital_one.pickle"
        if not Path(_capital_one_file_name).exists():
            self.save_capital_one_data()

    def get_data(self):
        return self._operations_account_data, self._credit_line_data, self._student_loan_data, self._capital_one_data

    def get_data_by_date(self, year=None, month=None, day=None):
        """
        This function returns transactions for all bank accounts for a specified time period. If year, month and day are
        not specified, defaults to all current year transactions
        args:
            - year:         year of the requested data, defaults to current year
            - month:        month of the requested data, defaults to current month
            - day:          day of the requested data, defaults to current day
        """
        _dfs_operations = pd.DataFrame()
        _dfs_credit_line = pd.DataFrame()
        _dfs_student_loan = pd.DataFrame()
        _dfs_capital_one = pd.DataFrame()

        if year is None:
            year = datetime.date.today().year
        _dfs_operations = self._operations_account_data[self._operations_account_data['date'].array.year == year]
        _dfs_credit_line = self._credit_line_data[self._credit_line_data['date'].array.year == year]
        _dfs_student_loan = self._student_loan_data[self._student_loan_data['date'].array.year == year]
        _dfs_capital_one = self._capital_one_data[self._capital_one_data['date'].array.year == year]

        if month is not None:
            _dfs_operations = _dfs_operations[_dfs_operations['date'].array.month == month]
            _dfs_credit_line = _dfs_credit_line[_dfs_credit_line['date'].array.month == month]
            _dfs_student_loan = _dfs_student_loan[_dfs_student_loan['date'].array.month == month]
            _dfs_capital_one = _dfs_capital_one[_dfs_capital_one['date'].array.month == month]

        # if day is none we get transaction for the whole month
        if day is not None:
            _dfs_operations = _dfs_operations[_dfs_operations['date'].array.day == day]
            _dfs_credit_line = _dfs_credit_line[_dfs_credit_line['date'].array.day == day]
            _dfs_student_loan = _dfs_student_loan[_dfs_student_loan['date'].array.day == day]
            _dfs_capital_one = _dfs_capital_one[_dfs_capital_one['date'].array.day == day]

        return _dfs_operations, _dfs_credit_line, _dfs_student_loan, _dfs_capital_one

    def _load_desjardins_data(self):
        '''
        Load Desjardins data from pickle object or csv file
        '''
        _desjardins_file_name = self._pickle_objects_root + "desjardins.pickle"

        if Path(_desjardins_file_name).exists():
            _df_file = open(_desjardins_file_name, 'rb')
            self._raw_desjardins_data = pickle.load(_df_file)
        else:
            if self._desjardins_input_path != "":
                self._raw_desjardins_data = _load_desjardins_csv_files(self._desjardins_input_path)

        self._raw_desjardins_data = self._raw_desjardins_data.sort_values(by=['account', 'date', 'transaction_num'])

    def _load_capital_one_data(self):
        '''
        Load Capital One data from pickle object or csv file
        '''
        _capital_one_file_name = self._pickle_objects_root + "capital_one.pickle"

        if Path(_capital_one_file_name).exists():
            _df_file = open(_capital_one_file_name, 'rb')
            self._raw_capital_one_data = pickle.load(_df_file)
        else:
            if self._desjardins_input_path != "":
                self._raw_capital_one_data = _load_capital_one_csv_files(self._capital_one_input_path)

        self._raw_capital_one_data = self._raw_capital_one_data.sort_values(by='date')

    def update_desjardins_data(self, input_path):
        '''
        Update Desjardins data from csv file
        '''
        if input_path == '':
            raise ValueError('please specify a path for Desjardins CSV files')

        # Desjardins csv files columns names
        names = ['branch', 'foliot', 'account', 'date', 'transaction_num', 'description', 'fees', 'withdrawal',
                 'deposit', 'interests', 'capital_paid', 'advance', 'reimboursment', 'balance']

        csv_files = Path(input_path).glob('*.csv')
        _cash_flow = pd.DataFrame(columns=names)
        for x in csv_files:
            if x.is_file():
                _cash_flow = _cash_flow.append(pd.read_csv(x.as_posix(),
                                                           encoding='latin1',
                                                           names=names),
                                               ignore_index=True)

        # Convert strings to actual date time objects
        _cash_flow['date'] = pd.to_datetime(_cash_flow['date'], format='%Y-%m-%d')
        _cash_flow = _cash_flow.replace(np.nan, 0)

        self._raw_desjardins_data = pd.concat([self._raw_desjardins_data, _cash_flow], axis=0)
        self._raw_desjardins_data.drop_duplicates(keep='first', subset=names, inplace=True)
        self._raw_desjardins_data.sort_values(by='date', inplace=True)

    def update_desjardins_data_from_clipboard(self):
        '''
        Update Desjardins data from clipboard-contained information (when copying data that is not yet on a statement
        from Desjardins)
        '''

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

        except:
            print(colorama.Fore.YELLOW + "Clipboard content is not recognized..." + colorama.Fore.RESET)

    def update_capital_one_data(self, input_path):
        '''
        Update Capital One data from csv file
        '''

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
        del self._raw_capital_one_data.sort_values['posted_date']

    def save_desjarding_data(self):
        '''
        Save Desjardins data as pickle object
        '''
        Path(self._pickle_objects_root).mkdir(exist_ok=True)
        _desjardins_file_name = self._pickle_objects_root + "desjardins.pickle"
        _df_file = open(_desjardins_file_name, 'wb')
        pickle.dump(self._raw_desjardins_data, _df_file, pickle.HIGHEST_PROTOCOL)

    def save_capital_one_data(self):
        '''
        Save Capital One data as pickle object
        '''
        Path(self._pickle_objects_root).mkdir(exist_ok=True)
        _capital_one_file_name = self._pickle_objects_root + "capital_one.pickle"
        _df_file = open(_capital_one_file_name, 'wb')
        pickle.dump(self._raw_capital_one_data, _df_file, pickle.HIGHEST_PROTOCOL)

    def update_desjardins_codes(self):
        self._raw_desjardins_data[self._raw_desjardins_data]