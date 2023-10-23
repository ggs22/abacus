import datetime
import logging
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import colorama
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from accounting import Account, PREDICTED_BALANCE


class ForecastStrategy:

    def get_serialized_prediction_path(self, account: Account, simulation_date: str) -> Path:
        pred_path = Path(account.conf.serialized_objects_dir)
        file_name = f"{account.name}_{self.__class__.__name__}_"
        if simulation_date == "":
            file_name += f"{account.most_recent_date.item().strftime('%Y-%m-%d')}"
        else:
            file_name += f"{simulation_date}"
        file_name += "_prediction.pkl"
        pred_path = pred_path.joinpath(file_name)
        return pred_path

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f"{self.__class__.__name__}")

    @staticmethod
    def _append_amount(account: Account, pred: dict, amount: float | Iterable[float]) -> None:
        if not isinstance(amount, Iterable):
            amount = [amount]
        for _amount in amount:
            neg = _amount <= 0
            pred[account.negative_names[0]].append(_amount * neg + 0 * (not neg))
            pred[account.positive_names[0]].append(0 * neg + _amount * (not neg))

    def _prediction_wraper(self,
                           account: Account,
                           predict_func: callable,
                           predicted_days: int = 365,
                           average_over: int = 365,
                           simulation_date: str = "",
                           force_new: bool = False,
                           **kwargs) -> pd.DataFrame | None:

        if account.status == "OPEN":

            # if not account.serialized_prediction_path.exists() or force_new:
            if not self.get_serialized_prediction_path(account, simulation_date).exists() or force_new:
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
                kwargs['end_date'] = period_end_date
                kwargs['sampling_period'], _ = account.get_period_data(period_start_date.strftime('%Y-%m-%d'),
                                                                       period_end_date.strftime('%Y-%m-%d'))

                pred: pd.DataFrame | None = predict_func(**kwargs)

                if pred is not None:
                    current_balance = account.current_balance

                    pred['date'] = pd.to_datetime(pred['date'].apply(lambda i: period_end_date + datetime.timedelta(days=i)))

                    pred.sort_values(by=['description', 'date'], ascending=True, ignore_index=True, inplace=True)
                    pred = pred.replace(np.nan, 0)
                    pred['balance'] = pred.groupby(by='description').cumsum(numeric_only=True).sum(axis=1) + current_balance

                    pred.reset_index(drop=True, inplace=True)
                    with open(self.get_serialized_prediction_path(account, simulation_date), 'wb') as pred_file:
                        pickle.dump(pred, pred_file)
            else:

                with open(self.get_serialized_prediction_path(account, simulation_date), 'rb') as pred_file:
                    pred = pickle.load(pred_file)
                self.logger.info(f"loaded {self.get_serialized_prediction_path(account, simulation_date)}")
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

            if simulation_date == "":
                simulation_date = account.most_recent_date.item().strftime("%Y-%m-%d")
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


class ForecastByMeanStrategy(ForecastStrategy):

    def predict(self,
                predicted_days: int,
                account: Account,
                average_over: int = 365,
                simulation_date: str = "",
                **kwargs) -> pd.DataFrame:

        def _predict(predicted_days: int,
                     stats: pd.DataFrame,
                     **kwargs) -> pd.DataFrame:

            daily_expense = stats['daily_mean'].mean()
            pred_l = {col_name: list() for col_name in account.columns_names}
            with tqdm(total=predicted_days,
                      desc=f"{self.__class__.__name__} iterating {account.name}") as pbar:
                for days_ix in range(predicted_days):
                    amount = daily_expense
                    pred_l['date'].append(days_ix + 1)
                    pred_l['description'].append(PREDICTED_BALANCE)
                    pred_l['code'].append("other")
                    self._append_amount(account=account, pred=pred_l, amount=amount)
                    pbar.update(1)

            self._prune_dict(pred_l)
            pred_l = pd.DataFrame(pred_l)

            return pred_l

        kwargs['strategy_name'] = self.__class__.__name__
        pred: pd.DataFrame = self._prediction_wraper(account=account,
                                                     predicted_days=predicted_days,
                                                     predict_func=_predict,
                                                     average_over=average_over,
                                                     simulation_date=simulation_date,
                                                     **kwargs)
        account.prediction = pred

        return pred


class PlannedTransactionsStrategy(ForecastStrategy):

    def predict(self,
                predicted_days: int,
                account: Account,
                average_over: int = 365,
                simulation_date: str = "",
                **kwargs) -> pd.DataFrame | None:

        planned_transations: pd.DataFrame | None = account.get_planned_transactions(start_date=simulation_date,
                                                                                    predicted_days=predicted_days)
        if planned_transations is not None:
            planned_codes = planned_transations.code.unique()
        else:
            planned_codes = list()

        def _predict(predicted_days: int,
                     stats: pd.DataFrame,
                     **kwargs) -> pd.DataFrame | None:

            pred_l = {col_name: list() for col_name in account.columns_names}
            with tqdm(total=predicted_days,
                      desc=f"{self.__class__.__name__} iterating {account.name}") as pbar:
                    daily_expense = stats[~stats.index.isin(planned_codes)]['daily_mean'].mean()

                    for days_ix in range(predicted_days):
                        amount = daily_expense if planned_transations is not None else 0
                        pred_l['date'].append(days_ix + 1)
                        pred_l['description'].append(PREDICTED_BALANCE)
                        pred_l['code'].append("other")
                        self._append_amount(account=account, pred=pred_l, amount=amount)
                        pbar.update(1)

            self._prune_dict(pred_l)
            pred_l = pd.DataFrame(pred_l)

            return pred_l

        kwargs['strategy_name'] = self.__class__.__name__
        pred: pd.DataFrame = self._prediction_wraper(account=account,
                                                     predicted_days=predicted_days,
                                                     predict_func=_predict,
                                                     average_over=average_over,
                                                     simulation_date=simulation_date,
                                                     **kwargs)

        if planned_transations is not None and account.status != "CLOSED":
            pred = pd.concat([planned_transations, pred], ignore_index=True)
            pred.sort_values(by='date', inplace=True)
            pred.reset_index(drop=True, inplace=True)
            pred['balance'] = (pred.loc[:, [account.positive_names[0], account.negative_names[0]]]).sum(axis=1).cumsum() + account.current_balance

        account.prediction = pred

        return pred


class DateBasedMonteCarloStrategy(ForecastStrategy):

    def predict(self,
                account: Account,
                predicted_days: int,
                average_over: int = 365,
                simulation_date: str = "",
                **kwargs) -> pd.DataFrame:

        def _predict(sampling_period: pd.DataFrame, end_date: datetime.date, mc_iterations: int = 100, **kwargs):

            if account.ignored_index is not None:
                sampling_period = sampling_period[~sampling_period.index.isin(account.ignored_index)]
            pred_l = {col_name: list() for col_name in account.columns_names}
            with tqdm(total=predicted_days * mc_iterations,
                      desc=f"{self.__class__.__name__} iterating {account.name}") as pbar:
                for mc_iteration in range(mc_iterations):
                    for days_ix, _ in enumerate(range(predicted_days)):
                        date = end_date + datetime.timedelta(days=days_ix + 1)
                        sampling_ix = (sampling_period['date'].dt.day == date.day) & \
                                      (sampling_period['date'].dt.month == date.month)
                        if sum(sampling_ix) > 0:
                            sampling_dates = sampling_period.loc[sampling_ix, 'date'].dt.date.unique()
                            sampling_date = random.choice(sampling_dates)
                            sample = sampling_period[sampling_period['date'].dt.date == sampling_date]
                            amount = (sample[account.numerical_names] * account.numerical_signs).sum(axis=1)

                            for _ in range(len(sample)):
                                pred_l['date'].append(days_ix + 1)
                                pred_l['description'].append(f"{PREDICTED_BALANCE}_{mc_iteration}")
                                pred_l['code'].append("other")
                            self._append_amount(account=account, pred=pred_l, amount=amount)
                        pbar.update(1)

                self._prune_dict(pred_l)
                pred_l = pd.DataFrame(pred_l)
            return pred_l

        kwargs['strategy_name'] = self.__class__.__name__
        pred = self._prediction_wraper(account=account,
                                       predicted_days=predicted_days,
                                       predict_func=_predict,
                                       average_over=average_over,
                                       simulation_date=simulation_date,
                                       **kwargs)
        account.prediction = pred

        return pred
