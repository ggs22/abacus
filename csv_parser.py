import pandas as pd
import pickle
import pathlib as path
import datetime


class CSV_Parser:
    def __init__(self, desjardins_input_path="", capital_one_input_path=""):

        if desjardins_input_path != "":
            self.desjardins_input_path = desjardins_input_path
            self._raw_desjardins_data = self._load_desjardins_csv_files(self.desjardins_input_path)

            self._operations_account_data = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'EOP',
                                            :]
            self._credit_line_data = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'MC2', :]
            self._student_loan_data = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'PR1', :]

            self._operations_account_data = self._operations_account_data.dropna(axis=1, how='all')
            self._credit_line_data = self._credit_line_data.dropna(axis=1, how='all')
            self._student_loan_data = self._student_loan_data.dropna(axis=1, how='all')

        if capital_one_input_path != "":
            self.capital_one_input_path = capital_one_input_path
            self._raw_capital_one_data = self._load_capital_one_csv_files(capital_one_input_path)
            self._capital_one_data = self._raw_capital_one_data.dropna(axis=1, how='all', inplace=False)

    def get_data(self):
        return self._operations_account_data, self._credit_line_data, self._student_loan_data, self._capital_one_data

    def get_data_by_date(self, year=None, month=None, day=None):
        """
        This function returns transactions for all bank accounts for a specified time period


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

        if month is None:
            month = datetime.date.today().month
        _dfs_operations = _dfs_operations[_dfs_operations['date'].array.month == month]
        _dfs_credit_line = _dfs_credit_line[_dfs_credit_line['date'].array.month == month]
        _dfs_student_loan = _dfs_student_loan[_dfs_student_loan['date'].array.month == month]
        _dfs_capital_one = _dfs_capital_one[_dfs_capital_one['date'].array.month == month]

        if day is not None:
            day = datetime.date.today().day
            _dfs_operations = _dfs_operations[_dfs_operations['date'].array.day == day]
            _dfs_credit_line = _dfs_credit_line[_dfs_credit_line['date'].array.day == day]
            _dfs_student_loan = _dfs_student_loan[_dfs_student_loan['date'].array.day == day]
            _dfs_capital_one = _dfs_capital_one[_dfs_capital_one['date'].array.day == day]

        return _dfs_operations, _dfs_credit_line, _dfs_student_loan, _dfs_capital_one

    def _load_desjardins_csv_files(self, input_path) -> pd.DataFrame:
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

        csv_files = path.Path(input_path).glob('*.csv')
        _cash_flow = pd.DataFrame(columns=names)
        for x in csv_files:
            if x.is_file():
                _cash_flow = _cash_flow.append(pd.read_csv(x.as_posix(),
                                                           encoding='latin1',
                                                           names=names),
                                               ignore_index=True)

        # Convert strings to actual date time objects
        _cash_flow['date'] = pd.to_datetime(_cash_flow['date'])

        # Adds column,and inputs transaction code
        _cash_flow['code'] = self._get_codes(_cash_flow['description'])

        return _cash_flow

    def _load_capital_one_csv_files(self, input_path) -> pd.DataFrame:
        """
        This function returns an aggloromated dataframe from capital one csv files

        args:
            - input_path        location of csv files to be agglomerated
        """

        if input_path == '':
            raise ValueError('please specify a path for Capital One CSV files')

        # Capital One csv files columns names
        names = ['date', 'posted_date', 'card_num', 'description', 'category', 'debit', 'credit']

        csv_files = path.Path(input_path).glob('*.csv')
        _cash_flow = pd.DataFrame(columns=names)
        for x in csv_files:
            if x.is_file():
                _cash_flow = _cash_flow.append(pd.read_csv(x.as_posix(), encoding='latin1', names=names),
                                               ignore_index=True)

        # Convert strings to actual date time objects
        _cash_flow['date'] = pd.to_datetime(_cash_flow['date'])
        _cash_flow['code'] = self._get_codes(_cash_flow['description'])

        del _cash_flow['posted_date']

        print(_cash_flow[_cash_flow['code'] == 'na']['description'].unique())

        for i in _cash_flow[_cash_flow['code'] == 'na'].iterrows():
            print(i)

        return _cash_flow

    def _get_codes(self, descriptions) -> pd.Series:
        """
        This function returns a vector corresponding to all transaction code associated to the description vector given
        as argument

        args:
            - description   A vector of transaction description (often reffered to as "object, item, description, etc."
                            in bank statements). The relations between the descriptions and the codes are contained in
                            "assignations.csv"
        """
        assignations = pd.read_csv('assignations.csv', encoding='utf-8', sep=',').dropna(axis=1, how='all')
        codes = list()
        for description in descriptions:
            codes.append("na")
            for col in assignations.iteritems():
                for row in col[1].dropna(axis=0):
                    # if (col[1].dropna(axis=0).str.lower().str.find(description.lower()) != -1).any():
                    if (description.lower().find(row.lower()) != -1):
                        codes[len(codes) - 1] = col[0]
                        break

        return pd.Series(codes)
