import datetime
import os

import numpy as np
import pandas as pd

from accounts.Accounts import AccountMetadata, get_common_codes
from accounts.Account import Account


class CIBC(Account):
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

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['credit'] - d['debit']
            d.drop(columns=['credit', 'debit'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):
        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if range_data.shape[0] > 0:
            withdrawals = range_data.loc[range_data.debit > 0, :].copy()
            deposits = range_data.loc[range_data.credit > 0, :].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'credit') | (columns == 'code')
            withdrawals_ix = (columns == 'debit') | (columns == 'code')
            withdrawals.drop(axis=1, columns=withdrawals.columns[~withdrawals_ix], inplace=True)
            deposits.drop(axis=1, columns=deposits.columns[~deposits_ix], inplace=True)

            withdrawals_counts = withdrawals.groupby(by='code').count()
            deposits_counts = deposits.groupby(by='code').count()

            withdrawals_freqs = np.divide(withdrawals_counts, delta)
            deposits_freqs = np.divide(deposits_counts, delta)

            daily_freqs = withdrawals_freqs.join(other=deposits_freqs, how='outer').replace(np.nan, 0)

            withdrawals_mean = withdrawals.groupby(by='code').mean()
            deposits_mean = deposits.groupby(by='code').mean()

            means = withdrawals_mean.join(other=deposits_mean, how='outer').replace(np.nan, 0)

            withdrawals_std = withdrawals.groupby(by='code').std()
            deposits_std = deposits.groupby(by='code').std()

            std_devs = withdrawals_std.join(other=deposits_std, how='outer').replace(np.nan, 0)

            daily_means = np.multiply(means, daily_freqs)
            daily_std = np.multiply(std_devs, daily_freqs)

            daily_means.debit *= -1
            daily_freqs.debit *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])
        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

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
                cash_flow = cash_flow.append(tmp_df, ignore_index=True)

        # Convert strings to actual date time objects
        if 'date' in cash_flow.columns:
            cash_flow['date'] = pd.to_datetime(cash_flow['date'], format='%Y-%m-%d')

        # Adds column,and inputs transaction code
        cash_flow = cash_flow.replace(np.nan, 0)
        cash_flow.sort_values(by=['date'], inplace=True)
        cash_flow.reset_index(inplace=True, drop=True)
        cash_flow['credit'] = cash_flow['credit'].apply(lambda i: float(i))
        cash_flow['debit'] = cash_flow['debit'].apply(lambda i: float(i))
        cash_flow.drop_duplicates(subset=cash_flow.columns[cash_flow.columns != 'code'], inplace=True)
        for ix, row in cash_flow[cash_flow['code'] == 0].iterrows():
            cash_flow.loc[ix, 'code'] = get_common_codes(pd.DataFrame(row).T).values[0]
            if cash_flow.loc[ix, 'code'] == "na":
                cash_flow.loc[ix, 'code'] = self.get_account_specific_codes(pd.DataFrame(row).T).values[0]

        new_entries = cash_flow.shape[0] - old_entries_num
        if new_entries > 0:
            self.transaction_data = cash_flow
            print(f'{new_entries} new entries added')
            if self._save_prompt():
                self.to_pickle()

    def get_predicted_balance(self, days: int = 90) -> pd.DataFrame:
        pass