import datetime
import os
from datetime import timedelta

import numpy as np
import pandas as pd

from accounting.Account import AccountMetadata
from accounting.Account import Account, get_common_codes, _is_planned_transaction


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

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['deposit'] - d['withdrawal']
            d.drop(columns=['withdrawal', 'deposit', 'balance', 'transaction_num'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):

        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if range_data.shape[0] > 0:
            withdrawals = range_data.loc[range_data.withdrawal > 0, :].copy()
            deposits = range_data.loc[range_data.deposit > 0, :].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'deposit') | (columns == 'code')
            withdrawals_ix = (columns == 'withdrawal') | (columns == 'code')
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

            daily_means.withdrawal *= -1
            daily_freqs.withdrawal *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])
        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

    def update_from_raw_files(self):
        new_entries = 0
        csv_files = os.listdir(self.metadata.raw_files_dir)
        for x in csv_files:
            if x[-4:] == '.csv':
                new = pd.read_csv(filepath_or_buffer=os.path.join(self.metadata.raw_files_dir, x),
                                  encoding='latin1',
                                  names=self.metadata.columns_names)
                for ix, row in new.iterrows():
                    # TODO transaction number-based duplication detection logic is not valid for
                    #  partial monthly statement downloads
                    if (row['date'], row['transaction_num']) in self.transaction_data.set_index(
                            keys=['date', 'transaction_num']).index:
                        continue
                    else:
                        if row['account'] == 'EOP':
                            row['date'] = pd.to_datetime(row['date'], format='%Y-%m-%d')
                            row['code'] = get_common_codes(pd.DataFrame(row).T).values[0]
                            if row['code'] == "na":
                                row['code'] = self.get_account_specific_codes(pd.DataFrame(row).T).values[0]
                            row.replace(np.nan, 0, inplace=True)
                            self.transaction_data = pd.concat([self.transaction_data, pd.DataFrame(row).T],
                                                              ignore_index=True)
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