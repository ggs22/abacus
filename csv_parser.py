import pandas as pd
import pickle
import pathlib as path
import datetime

class CashFLow(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        super(CashFLow, self).__init__(args, kwargs)

    def get_data_by_date(self, year=None, month=None, day=None):
        _df = pd.DataFrame()
        if year is None:
            year = datetime.date.today().year
        _df = self[self['date'].array.year == year]

        if month is None:
            month = datetime.date.today().month
        _df = _df[_df['date'].array.month == month]

        if day is None:
            day = datetime.date.today().day
            _df = _df[_df['date'].array.day == day]

        return _df

class CSV_Parser:
    def __init__(self, desjardins_input_path="", capital_one_input_path=""):

        if desjardins_input_path != "":
            self.desjardins_input_path = desjardins_input_path
            self._raw_desjardins_data = self._load_desjardins_csv_files(self.desjardins_input_path)

            self._operations_account_data = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'EOP',
                                            :]
            self._credit_line_data = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'MC2', :]
            self._student_loan_data = self._raw_desjardins_data.loc[self._raw_desjardins_data['account'] == 'PR1', :]

            self._operations_account_data.dropna(axis=1, how='all', inplace=True)
            self._credit_line_data.dropna(axis=1, how='all', inplace=True)
            self._student_loan_data.dropna(axis=1, how='all', inplace=True)

        if capital_one_input_path != "":
            self.capital_one_input_path = capital_one_input_path
            self._raw_capital_one_data = self._load_capital_one_csv_files(capital_one_input_path)
            self._capital_one_data = self._raw_capital_one_data.dropna(axis=1, how='all', inplace=True)

    def get_data(self):
        return self._operations_account_data, self._credit_line_data, self._student_loan_data, self._capital_one_data

    def _load_desjardins_csv_files(self, input_path) -> CashFLow:
        """
        This function returns an aggloromated dataframe from dejardins csv files

        args:
            - input_path        location of csv files to be agglomerated
        """

        if input_path == '':
            raise ValueError('please specify a path for Desjardins CSV files')

        # Desjardins csv files columns names
        names = ['branch', 'foliot', 'account', 'date', 'transaction_num', 'Object', 'fees',
                 'withdrawal', 'deposit', 'interests', 'capital_paid', 'advance', 'reimboursment', 'balance']

        csv_files = path.Path(input_path).glob('*.csv')
        _cash_flow = CashFLow(columns=names)
        for x in csv_files:
            if x.is_file():
                _cash_flow = _cash_flow.append(pd.read_csv(x.as_posix(), encoding='latin1', header=0, names=names), ignore_index=True)

        # Convert strings to actual date time objects
        _cash_flow['date'] = pd.to_datetime(_cash_flow['date'])

        return _cash_flow

    def _load_capital_one_csv_files(self, input_path) -> CashFLow:
        """
        This function returns an aggloromated dataframe from capital one csv files

        args:
            - input_path        location of csv files to be agglomerated
        """

        if input_path == '':
            raise ValueError('please specify a path for Capital One CSV files')

        # Capital One csv files columns names
        names = ['transaction_date', 'posted_date', 'card_num', 'description', 'category', 'debit', 'credit']

        csv_files = path.Path(input_path).glob('*.csv')
        _cash_flow = CashFLow(columns=names)
        for x in csv_files:
            if x.is_file():
                _cash_flow = _cash_flow.append(pd.read_csv(x.as_posix(), encoding='latin1', header=0, names=names), ignore_index=True)

        # Convert strings to actual date time objects
        _cash_flow['transaction_date'] = pd.to_datetime(_cash_flow['transaction_date'])
        del _cash_flow['posted_date']

        return _cash_flow



