import datetime
import os

import numpy as np
import pandas as pd

from accounting.Account import AccountMetadata
from accounting.Account import Account, get_common_codes


class PayPal(Account):
    def __init__(self, lmetadata: AccountMetadata):
        super().__init__(lmetadata=lmetadata)
        self.col_mask = ['date', 'description', 'currency', 'amount', 'balance', 'code']

    # TTODO refactor to load data in parent class and process it
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

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['amount']
            d.drop(columns=['amount', 'balance'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)
        if range_data.shape[0] > 0:

            withdrawals = range_data[range_data.amount < 0].copy()
            deposits = range_data[range_data.amount >= 0].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'amount') | (columns == 'code')
            withdrawals_ix = (columns == 'amount') | (columns == 'code')
            withdrawals.drop(axis=1, columns=withdrawals.columns[~withdrawals_ix], inplace=True)
            deposits.drop(axis=1, columns=deposits.columns[~deposits_ix], inplace=True)

            withdrawals_counts = withdrawals.groupby(by='code').count()
            deposits_counts = deposits.groupby(by='code').count()

            withdrawals_freqs = np.divide(withdrawals_counts, delta)
            deposits_freqs = np.divide(deposits_counts, delta)

            daily_freqs = withdrawals_freqs.join(other=deposits_freqs, how='outer', lsuffix='_out').replace(np.nan, 0)

            withdrawals_mean = withdrawals.groupby(by='code').mean()
            deposits_mean = deposits.groupby(by='code').mean()

            means = withdrawals_mean.join(other=deposits_mean, how='outer', lsuffix='_out').replace(np.nan, 0)

            withdrawals_std = withdrawals.groupby(by='code').std()
            deposits_std = deposits.groupby(by='code').std()

            std_devs = withdrawals_std.join(other=deposits_std, how='outer', lsuffix='_out').replace(np.nan, 0)

            daily_means = np.multiply(means, daily_freqs)
            daily_std = np.multiply(std_devs, daily_freqs)

            daily_freqs.amount_out *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])
        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

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
        cash_flow['code'] = get_common_codes(cash_flow)
        l1 = self.transaction_data.shape[0]
        self.transaction_data = self.transaction_data.append(cash_flow)
        self.transaction_data.drop_duplicates(inplace=True)
        self.transaction_data.reset_index(inplace=True, drop=True)
        l2 = self.transaction_data.shape[0]

        if l2 - l1 > 0:
            print(f'{l2 - l1} new entries added')
            if self._save_prompt():
                self.to_pickle()

    # TODO implement prediction method
    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        df = self.get_data()
        return df