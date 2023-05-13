import datetime
import os
import pickle
import random
from datetime import timedelta
from typing import List, Union

import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt, dates as mdates
from tqdm import tqdm

from accounts.Accounts import AccountMetadata, get_common_codes, _is_planned_transaction
from accounts.Account import Account
from utils.utils import pickle_dir


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

    def get_summed_date_range_data(self, start_date: datetime.date, end_date: datetime.date):
        d = super().get_summed_date_range_data(start_date=start_date, end_date=end_date)
        if 'internal_cashflow' in d.index:
            d.drop(labels=['internal_cashflow'], inplace=True)
        if d.shape[0] > 0:
            d['total'] = d['interests'].apply(lambda i: -i)
            d.drop(columns=['transaction_num', 'interests', 'advance', 'reimboursment', 'balance'], inplace=True)
        else:
            d = None
        return d

    def get_date_range_daily_average(self, start_date: datetime.date, end_date: datetime.date):

        # calculate the number of days in the date range
        delta = (end_date - start_date).days

        # get data and separate deposits from withdrawals
        range_data = self.get_data_by_date_range(start_date=start_date, end_date=end_date)

        if range_data.shape[0] > 0:
            withdrawals = range_data.loc[(range_data.advance > 0) | (range_data.interests > 0), :].copy()
            deposits = range_data.loc[range_data.reimboursment > 0, :].copy()

            columns = withdrawals.columns
            deposits_ix = (columns == 'reimboursment') | (columns == 'code')
            withdrawals_ix = (columns == 'interests') | (columns == 'advance') | (columns == 'code')
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

            daily_means.interests *= -1
            daily_means.advance *= -1
            daily_freqs.interests *= -1
            daily_freqs.advance *= -1

            daily_means = pd.DataFrame(daily_means.sum(axis=1), columns=['total_mean'])
            daily_std = pd.DataFrame(daily_std.sum(axis=1), columns=['total_std'])
            daily_freqs = pd.DataFrame(daily_freqs.sum(axis=1), columns=['total_freq'])
        else:
            daily_means, daily_std, daily_freqs = None, None, None

        return daily_means, daily_std, daily_freqs

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
                            row['code'] = get_common_codes(pd.DataFrame(row).T).values[0]
                            if row['code'] == "na":
                                row['code'] = self.get_account_specific_codes(pd.DataFrame(row).T).values[0]
                            row.replace(np.nan, 0, inplace=True)
                            self.transaction_data = self.transaction_data.append(other=row, ignore_index=True)
                            new_entries += 1

        if new_entries > 0:
            self.transaction_data.drop_duplicates(keep='first', subset=self.metadata.columns_names, inplace=True)
            self.transaction_data.sort_values(by=['date', 'transaction_num'], inplace=True)

            print(f'{new_entries} new entries added')
            if self._save_prompt():
                self.to_pickle()

    def get_predicted_balance(self,
                              get_avg_method: callable,
                              end_date: datetime.date,
                              sim_date: datetime.date = None,
                              force_new: bool = False,
                              avg_interval: int = 90,
                              montecarl_iterations=3) -> pd.DataFrame:
        """
        This function gives an estimation of the future balance of the account for a specified number of days. It
        uses the planned transaction data and optionally the average spending data.
        :param get_avg_method a method to calculate daily averages with the arguments
               (start_date: datetime.Date, end_date=datetime.date)
        :param end_date: date at which the prediction ends
        :param sim_date: date at which the prediction starts
        :param avg_interval: The number of days over which the average and stdev are computed for probabilistic calcs.
        :return: a Dataframe containing the predicted balance of the account
        """

        fname = f'prd' + f'_{sim_date}' * (sim_date is not None) + f'_{end_date}.pkl'
        fpath = os.path.join(pickle_dir, fname)
        if os.path.exists(fpath) and not force_new:
            print(f'Prediction loaded from previous execution ({fpath})')
            with open(fpath, 'rb') as file:
                df = pickle.load(file=file)
        else:
            # if no sim_date is specified, the prediction starts from the last entry of transaction data
            df = self.get_data().copy()
            if sim_date is None:
                idate = pd.to_datetime(df.tail(1)['date'].values).date[0]
            else:
                if sim_date > end_date:
                    raise RuntimeError(f'Simulation starting date ({sim_date}) is ulterior to simulation ending date '
                                       f'({end_date})')
                idate = sim_date
                df.drop(labels=df[df['date'] > str(sim_date)].index, inplace=True)

            # Get the average, std dev. and daily freqencies of transaction for the specified time interval
            sdate = idate - datetime.timedelta(days=avg_interval)
            avg, std, freqs = get_avg_method(start_date=sdate, end_date=idate)

            balances_list = [df.tail(1).balance.values[0]].copy()
            initial_bal_value = balances_list[0]

            # Monte carlo iterations
            t_num_mc_prefix = 0
            digit_num = int(np.log10(montecarl_iterations)) + 1
            for mc_iteration in tqdm(range(0, montecarl_iterations),
                                     position=0,
                                     leave=False,
                                     desc='Monte-Carlo iterations'):

                # we are going to accumulate MC iterations values in this dict
                transaction_tuples = {'date': list(),
                                      'transaction_num': list(),
                                      'description': list(),
                                      'interests': list(),
                                      'advance': list(),
                                      'reimboursment': list(),
                                      'balance': list(),
                                      'code': list()}

                # for each day in the projection
                for delta_days in tqdm(range(1, (end_date-idate).days),
                                       position=1,
                                       leave=False,
                                       desc='Projected days'):
                    date = (idate + timedelta(days=delta_days))
                    t_num_date_prefix = 0

                    # add planned transaction tuple
                    for ix, ptransaction in self.planned_transactions.iterrows():
                        if _is_planned_transaction(year=ptransaction['year'],
                                                   month=ptransaction['month'],
                                                   day=ptransaction['day'],
                                                   date=date):

                            transaction_tuples['date'].append(pd.to_datetime(date))
                            transaction_tuples['transaction_num'].append(f'{t_num_mc_prefix:0{digit_num}d}_'
                                                                         f'{t_num_date_prefix:03d}_'
                                                                         f'planned')
                            transaction_tuples['description'].append(ptransaction['description'])
                            transaction_tuples['interests'].append(0)
                            transaction_tuples['advance'].append(ptransaction['withdrawal'])
                            transaction_tuples['reimboursment'].append(ptransaction['deposit'])
                            balance = balances_list[-1:][0] + ptransaction['withdrawal'] - ptransaction['deposit']
                            transaction_tuples['balance'].append(balance)
                            transaction_tuples['code'].append(ptransaction['code'])

                            balances_list.append(balance)
                            t_num_date_prefix += 1

                    # add average expenses to prediction
                    for expense_code in avg.index:
                        if expense_code.lower() not in ['internet', 'rent', 'pay',
                                                        'cell', 'hydro', 'interest', 'other',
                                                        'credit', 'impots & tps', 'trading']:

                            # compute probabilistic threshold of transaction occurence
                            amount = random.gauss(mu=avg.loc[expense_code, 'total_mean'] / freqs.loc[expense_code, 'total_freq'],
                                                  sigma=std.loc[expense_code, 'total_std'] / freqs.loc[expense_code, 'total_freq'])
                            amount *= np.sign(avg.loc[expense_code, 'total_mean'])
                            amount = round(amount, 2)
                            thresh_prob = random.uniform(0, 1)
                            event_prob = abs(freqs.loc[expense_code, 'total_freq'])
                            amount = amount * (thresh_prob < event_prob)
                            amount = min(amount, 0) * (avg.loc[expense_code, 'total_mean'] < 0) + \
                                     max(amount, 0) * (avg.loc[expense_code, 'total_mean'] >= 0)

                            if amount != 0:
                                transaction_tuples['date'].append(pd.to_datetime(date))
                                transaction_tuples['transaction_num'].append(f'{t_num_mc_prefix:0{digit_num}d}_'
                                                                             f'{t_num_date_prefix:03d}_'
                                                                             f'mc_expected')
                                transaction_tuples['description'].append(expense_code)
                                transaction_tuples['interests'].append(0)
                                transaction_tuples['advance'].append(
                                    abs(amount) * (avg.loc[expense_code, 'total_mean'] < 0))
                                transaction_tuples['reimboursment'].append(
                                    abs(amount) * (avg.loc[expense_code, 'total_mean'] >= 0))
                                balance = balances_list[-1:][0] - transaction_tuples['reimboursment'][-1:][0] + \
                                          transaction_tuples['advance'][-1:][0]
                                transaction_tuples['balance'].append(balance)
                                transaction_tuples['code'].append(expense_code)

                                balances_list.append(balance)
                                t_num_date_prefix += 1

                            # in the first iteration we want to compute the non-conditionnal average (will be the dotted line)
                            if mc_iteration == 0:
                                amount = round(avg.loc[expense_code, 'total_mean'], 2)
                                transaction_tuples['date'].append(pd.to_datetime(date))
                                transaction_tuples['transaction_num'].append(f'{t_num_mc_prefix:0{digit_num}d}_'
                                                                             f'{t_num_date_prefix:03d}_'
                                                                             f'mean_expected')
                                transaction_tuples['description'].append(expense_code)
                                transaction_tuples['interests'].append(0)
                                transaction_tuples['advance'].append(abs(amount) * (avg.loc[expense_code, 'total_mean'] < 0))
                                transaction_tuples['reimboursment'].append(abs(amount) * (avg.loc[expense_code, 'total_mean'] >= 0))
                                balance = balances_list[-1:][0] - transaction_tuples['reimboursment'][-1:][0] + transaction_tuples['advance'][-1:][0]
                                transaction_tuples['balance'].append(balance)
                                transaction_tuples['code'].append(expense_code)

                                balances_list.append(balance)
                                t_num_date_prefix += 1

                if montecarl_iterations > 1:
                    # insert montecarlo reset tuple
                    transaction_tuples['date'].append(pd.to_datetime(idate))
                    transaction_tuples['transaction_num'].append(f'montecarlo_reset')
                    transaction_tuples['description'].append(expense_code)
                    transaction_tuples['interests'].append(0)
                    transaction_tuples['advance'].append(1)
                    transaction_tuples['reimboursment'].append(1)
                    balance = initial_bal_value
                    transaction_tuples['balance'].append(balance)
                    transaction_tuples['code'].append('na')

                    balances_list.append(initial_bal_value)
                    t_num_mc_prefix += 1

                df_pred = pd.DataFrame(transaction_tuples)

                # add interest estimate
                df_pred.sort_values(by=['date', 'transaction_num'], inplace=True)
                df_pred['delta'] = df_pred.date.diff().shift(-1)
                df_pred['delta'] = df_pred.delta.array.days
                df_pred['cap_interest'] = np.multiply(df_pred.balance, df_pred.delta) * self.metadata.interest_rate / 365
                df_pred.loc[df_pred['cap_interest'] < 0, 'cap_interest'] = 0
                df_pred.replace(np.nan, 0, inplace=True)
                df_pred.reset_index(drop=True, inplace=True)

                interest_sum = 0
                itt = iter(df_pred.iterrows())
                for ix, row in itt:
                    interest_sum += row['cap_interest']
                    if row['date'].day == 1 and row['interests'] == 0 and row['code'] == 'interest':
                        while row['date'].day == 1:
                            if row['code'] == 'interest' and row['interests'] == 0:
                                df_pred.loc[ix, 'interests'] = interest_sum
                                df_pred.loc[ix, 'balance'] += interest_sum
                            ix, row = next(itt)
                        interest_sum = 0

                df = pd.concat([df, df_pred]).sort_values(by=['date', 'transaction_num'])
                df.dropna(axis=0, how='all', inplace=True)
                ix = (df['interests'] == 0) &\
                     (df['advance'] == 0) &\
                     (df['reimboursment'] == 0) &\
                     (df['transaction_num'] == "expected")
                df.drop(labels=df.index[ix], axis=0, inplace=True)
                df.sort_values(by=['date', 'transaction_num'], inplace=True)
                df.reset_index(drop=True, inplace=True)

            with open(os.path.join(pickle_dir, fname), 'wb') as file:
                pickle.dump(df, file=file)

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

    def plot_prediction_compare(self,
                                get_avg_method: callable,
                                end_date: datetime.date,
                                start_date: datetime.date = None,
                                sim_dates: List[Union[datetime.date, None]] = None,
                                show=False,
                                fig_size=(7, 8),
                                force_new: bool = False,
                                avg_interval: int = 90,
                                montecarl_iterations=100) -> None:
        """

        :param end_date:        The simulation end date.
        :param start_date:      The simulation start date.
        :param sim_dates:       A list of dates from which a simulation starts even though actual data follows that date
        :param show:            Show the plot, as opposed to returning only the canvas.
        :param fig_size:        The size of the figure in inches.
        :param force_new:       Force a new simulation even though a preceding simulation has been run wih the
                                same dates.
        :param avg_interval:    The interval of days preceeding the simulation start over which the mean and std dev
                                are computed for each transaction code.
        """

        def plot_predictions(predictions: List[pd.DataFrame]):
            for ix, pred in enumerate(tqdm(predictions, desc='ploting predictions...')):
                fill_colors = (random.random(), random.random(), random.random())
                passed_ix = ~(pred["transaction_num"].str.contains("planned") |
                              (pred["transaction_num"].str.contains("expected")) |
                              (pred["transaction_num"].str.contains("montecarlo")))

                mc_futur_ix = ~passed_ix & (pred["transaction_num"].str.contains("mc") | pred["transaction_num"].str.contains("montecarlo"))
                # mean_futur_ix = ~passed_ix & (pred["transaction_num"].str.contains("mean") | pred["transaction_num"].str.contains("montecarlo"))

                df_mean = pred[np.where(mc_futur_ix == True)[0][0] - 1:].copy().groupby(by='date').mean(numeric_only=True)
                df_std = pred[np.where(mc_futur_ix == True)[0][0] - 1:].copy().groupby(by='date').std(numeric_only=True)
                plt.fill_between(x=pd.to_datetime(df_std.index),
                                 y1=df_mean.balance - 3 * df_std.balance,
                                 y2=df_mean.balance + 3 * df_std.balance,
                                 color=fill_colors,
                                 alpha=0.45)
                plt.plot(pd.to_datetime(df_std.index),
                         df_mean.balance,
                         color=fill_colors,
                         linestyle='--')

                # we want to plot past transaction only once
                if ix == len(predictions) - 1:
                    # sns.lineplot(data=pred[passed_ix].copy(), x='date', y='balance', color='g')
                    plt.plot(pd.to_datetime(pred[passed_ix].date), pred[passed_ix].balance, color='g')

        predictions: List[pd.DataFrame] = list()

        if sim_dates is None:
            sim_dates = [None]
        else:
            sim_dates.append(None)

        for sdate in sim_dates:
            prediction = self.get_predicted_balance(get_avg_method=get_avg_method,
                                                    end_date=end_date,
                                                    sim_date=sdate,
                                                    force_new=force_new,
                                                    avg_interval=avg_interval,
                                                    montecarl_iterations=montecarl_iterations).copy()
            prediction.loc[:, 'balance'] = -1 * prediction.loc[:, 'balance']
            if start_date is None:
                start_date = datetime.datetime.today().date() - timedelta(days=7)
            prediction = prediction[prediction['date'].array.date > start_date]
            prediction.date = prediction.date.apply(func=lambda i: str(i).replace(' 00:00:00', ''))
            predictions.append(prediction)

        title = f'Prediction {start_date} to {end_date}'
        fig = plt.figure(figsize=fig_size, num=title)

        plot_predictions(predictions=predictions)

        plt.title(title)
        plt.grid(b=True, which='major', axis='both')
        plt.grid(b=True, which='minor', axis='x')
        plt.xticks(rotation=90)
        min_y = min([estim.balance.min() for estim in predictions])
        max_y = max([estim.balance.max() for estim in predictions])
        plt.yticks(ticks=np.arange(np.round(min_y - 1000, -3),
                                   np.round(max_y + 1000, -3),
                                   500))

        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        if show:
            plt.show()
        return fig

    def _get_last_interest_payment_date(self) -> datetime.date:
        return (self.transaction_data[self.transaction_data['interests'] > 0]).tail(n=1).date.array.date[0]

    def _get_last_interest_payment_index(self, sim_date: datetime.date = None) -> datetime.date:
        if sim_date is not None:
            res = (self.transaction_data[(self.transaction_data['interests'] > 0) &
                                         (self.transaction_data['date'].array.date < sim_date)]).tail(n=1).index[0]
        else:
            res = (self.transaction_data[self.transaction_data['interests'] > 0]).tail(n=1).index[0]
        return res

    def _get_next_interest_payment_date(self):
        d = self._get_last_interest_payment_date()
        d2 = d + relativedelta(months=+1)
        # no transactions on saturday or sunday, instead it is reported to next monday
        while d2.isoweekday() in [6, 7]:
            d2 = d2 + timedelta(days=1)
        return (self.transaction_data[self.transaction_data['interests'] > 0]).tail(n=1).date.array.date[0]