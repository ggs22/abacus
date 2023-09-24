from copy import deepcopy

import datetime
import random as r

import colorama
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from accounting.Account import Account, PREDICTED_BALANCE


class PredictionStrategy:

    @staticmethod
    def _append_amount(account: Account, pred: dict, amount: float) -> None:
        neg = amount <= 0
        pred[account.negative_names[0]].append(amount * neg + 0 * (not neg))
        pred[account.positive_names[0]].append(0 * neg + amount * (not neg))

    @staticmethod
    def _prediction_wraper(account: Account,
                           predict_func: callable,
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
            kwargs['stats'] = stats
            kwargs['predicted_days'] = predicted_days

            pred = predict_func(**kwargs)

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

    @staticmethod
    def _prune_dict(_dict: dict) -> None:
        keys_to_pop = list()
        for key, val in _dict.items():
            if len(val) == 0:
                keys_to_pop.append(key)
        for key in keys_to_pop:
            _dict.pop(key)

    def plot_prediction(self,
                        account: Account,
                        predicted_days: int = 365,
                        figure_name: str = "",
                        simulation_date: str = "",
                        **kwargs) -> None:

            pred: pd.DataFrame | None = self.predict(account=account,
                                                     predicted_days=predicted_days,
                                                     simulation_date=simulation_date,
                                                     **kwargs)

            if pred is not None:
                mean = pred.loc[:, ['date', 'balance']].groupby(by='date').mean()
                std = pred.loc[:, ['date', 'balance']].groupby(by='date').std()
                plt.figure(num=figure_name)
                plt.plot(mean,
                         label="",
                         linestyle='--',
                         c=account.color)
                plt.fill_between(x=mean.index,
                                 y1=(mean-std)['balance'],
                                 y2=(mean+std)['balance'],
                                 color=account.color,
                                 alpha=0.3)


class PredictionByMeanStrategy(PredictionStrategy):

    def predict(self,
                predicted_days: int,
                account: Account,
                average_over: int = 365,
                simulation_date: str = "",
                **kwargs) -> pd.DataFrame:

        def _predict(predicted_days: int,
                     stats: pd.DataFrame,
                     **kwargs):

            daily_expense = stats['daily_mean'].mean()
            pred_l = {col_name: list() for col_name in account.columns_names}
            for days_ix, _ in tqdm(enumerate(range(predicted_days))):
                amount = daily_expense
                pred_l['date'].append(days_ix + 1)
                pred_l['description'].append(PREDICTED_BALANCE)
                pred_l['code'].append("other")
                self._append_amount(account=account, pred=pred_l, amount=amount)

            self._prune_dict(pred_l)
            pred_l = pd.DataFrame(pred_l)

            return pred_l

        pred = self._prediction_wraper(account=account,
                                       predicted_days=predicted_days,
                                       predict_func=_predict,
                                       average_over=average_over,
                                       simulation_date=simulation_date,
                                       **kwargs)
        account.prediction = pred

        return pred


class BasicMonteCarloStrategy(PredictionStrategy):

    def predict(self,
                predicted_days: int,
                account: Account,
                average_over: int = 365,
                simulation_date: str = "",
                **kwargs) -> pd.DataFrame:

        def _predict(stats: pd.DataFrame, mc_iterations: int = 100, **kwargs):
            pred_l = {col_name: list() for col_name in account.columns_names}
            with tqdm(total=predicted_days * mc_iterations, desc=f"Monte-Carlo iterations {account.name}") as pbar:
                for mc_iteration in range(mc_iterations):
                    for days_ix, _ in enumerate(range(predicted_days)):
                        amount = r.gauss(mu=stats['daily_mean'].mean(), sigma=stats['daily_std'].std())
                        pred_l['date'].append(days_ix + 1)
                        pred_l['description'].append(f"{PREDICTED_BALANCE}_{mc_iteration}")
                        pred_l['code'].append("other")
                        self._append_amount(account=account, pred=pred_l, amount=amount)
                        pbar.update(1)

                self._prune_dict(pred_l)
                pred_l = pd.DataFrame(pred_l)
            return pred_l

        pred = self._prediction_wraper(account=account,
                                       predicted_days=predicted_days,
                                       predict_func=_predict,
                                       average_over=average_over,
                                       simulation_date=simulation_date,
                                       **kwargs)
        account.prediction = pred

        return pred
