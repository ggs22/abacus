from copy import deepcopy

import datetime
import random as r

import colorama
import numpy as np
import pandas as pd

from tqdm import tqdm

from accounting.Account import Account, PREDICTED_BALANCE


def _prune_dict(_dict: dict) -> None:
    keys_to_pop = list()
    for key, val in _dict.items():
        if len(val) == 0:
            keys_to_pop.append(key)
    for key in keys_to_pop:
        _dict.pop(key)


class PredictionStrategy:
    def __init__(self, stats: pd.DataFrame, account: Account, planned_transaction: pd.DataFrame):
        self.stats = stats
        self.account = account
        self.planned_transactions = planned_transaction

    def append_amount(self, df: pd.DataFrame, amount: float) -> None:
        neg = amount <= 0
        df[self.account.negative_names[0]].append(amount * neg + 0 * (not neg))
        df[self.account.positive_names[0]].append(0 * neg + amount * (not neg))


class PredictionByMeanStrategy(PredictionStrategy):

    def __init__(self, stats: pd.DataFrame, account: Account):
        super().__init__(stats=stats, account=account, planned_transaction=None)

    def predict(self, predicted_days: int) -> pd.DataFrame:
        daily_expense = self.stats['daily_mean'].mean()
        pred = {col_name: list() for col_name in self.account.columns_names}
        for days_ix, _ in tqdm(enumerate(range(predicted_days))):
            amount = daily_expense
            pred['date'].append(days_ix + 1)
            pred['description'].append(PREDICTED_BALANCE)
            pred['code'].append("other")
            self.append_amount(df=pred, amount=amount)

        _prune_dict(pred)
        pred = pd.DataFrame(pred)

        return pred


class BasicMonteCarloStrategy(PredictionStrategy):

    def __init__(self, stats: pd.DataFrame, account: Account):
        super().__init__(stats=stats, account=account, planned_transaction=None)

    def predict(self, predicted_days: int, mc_iterations: int = 100) -> pd.DataFrame:
        pred = {col_name: list() for col_name in self.account.columns_names}
        with tqdm(total=predicted_days*mc_iterations, desc=f"Monte-Carlo iterations {self.account.name}") as pbar:
            for mc_iteration in range(mc_iterations):
                for days_ix, _ in enumerate(range(predicted_days)):
                    amount = r.gauss(mu=self.stats['daily_mean'].mean(), sigma=self.stats['daily_std'].std())
                    pred['date'].append(days_ix + 1)
                    pred['description'].append(f"{PREDICTED_BALANCE}_{mc_iteration}")
                    pred['code'].append("other")
                    self.append_amount(df=pred, amount=amount)
                    pbar.update(1)

        _prune_dict(pred)
        pred = pd.DataFrame(pred)

        return pred


def get_balance_prediction(account: Account,
                           predicted_days: int = 365,
                           average_over: int = 365,
                           simulation_date: str = "",
                           **kwargs) -> pd.DataFrame | None:

    if account.status == "OPEN":
        if simulation_date == "":
            past_data: pd.DataFrame = deepcopy(
                account.transaction_data
            )
        else:
            simulation_date = datetime.date.fromisoformat(simulation_date)
            past_data: pd.DataFrame = deepcopy(
                account.transaction_data[account.transaction_data.date.array.date <= simulation_date]
            )
        period_end_date = past_data.tail(1).date.array.date[0]
        period_start_date = period_end_date - datetime.timedelta(days=average_over)

        stats = account.period_stats(period_start_date.strftime('%Y-%m-%d'),
                                     period_end_date.strftime('%Y-%m-%d'),
                                     **kwargs)

        mean_predictor = PredictionByMeanStrategy(stats=stats, account=account)
        mc_predictor = BasicMonteCarloStrategy(stats=stats, account=account)

        """chnage strategy here"""
        # pred = mean_predictor.predict(predicted_days=predicted_days)
        mc_iterations = kwargs['mc_iterations'] if 'mc_iterations' in kwargs else 100
        pred = mc_predictor.predict(predicted_days=predicted_days, mc_iterations=mc_iterations)

        # past_data['balance'] = account.balance_column.iloc[:, 0].to_numpy()

        current_balance = account.current_balance

        pred['date'] = pd.to_datetime(pred['date'].apply(lambda i: period_end_date + datetime.timedelta(days=i)))

        pred.sort_values(by=['description', 'date'], ascending=True, ignore_index=True, inplace=True)
        pred = pred.replace(np.nan, 0)
        pred['balance'] = pred.groupby(by='description').cumsum(numeric_only=True).sum(axis=1) + current_balance

        pred.reset_index(drop=True, inplace=True)

    else:

        print(colorama.Fore.YELLOW,
              f"No prediction for account {account.name}, status: {account.status} !",
              colorama.Fore.RESET,
              sep="")
        pred = None

    return pred
