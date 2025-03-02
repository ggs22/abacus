import datetime
import logging
import pickle
import random
from copy import deepcopy, copy
from pathlib import Path
from typing import Iterable, List, Any, Dict, Tuple

import colorama
import numpy as np
import pandas as pd
from tqdm import tqdm

from accounting import Account, AccountStats, PREDICTED_BALANCE


ITERATION = "iteration"


class ForecastStrategy:

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def get_serialized_prediction_path(self, account: Account, simulation_date: str) -> Path:
        pred_path = Path(account.serialized_self_path).parent
        file_name = f"{account.name}_{self.__class__.__name__}_"
        if simulation_date == "":
            file_name += f"{account.most_recent_date.item().strftime('%Y-%m-%d')}"
        else:
            file_name += f"{simulation_date}"
        file_name += "_prediction.pkl"
        pred_path = pred_path.joinpath(file_name)
        pred_path.parent.mkdir(exist_ok=True, parents=True)
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
            if account.positive_names[0] != account.negative_names[0]:
                pred[account.positive_names[0]].append(0 * neg + _amount * (not neg))

    def _prediction_wraper(self,
                           account: Account,
                           predict_func: callable,
                           stats: AccountStats | None = None,
                           simulation_date: str = "",
                           force_new: bool = False) -> pd.DataFrame | None:

        if account.status == "OPEN":

            # check if the prediction already exists
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

                # compute account's default stats. if no alternative statistics have been provided
                if stats is None:
                    stats = account.period_stats(date="", last_n_days=365)
                pred: pd.DataFrame | None = predict_func(stats)

                # if the prediction is not empty
                if pred is not None and pred.shape[0] > 0:

                    # convert ints to datetime object (ints are used to accelerate sims. iterations)
                    pred['date'] = pd.to_datetime(pred['date'].apply(lambda i: period_end_date + datetime.timedelta(days=i)))

                    with open(self.get_serialized_prediction_path(account, simulation_date), 'wb') as pred_file:
                        pickle.dump(pred, pred_file)

            # load the prediction if it had been serialized
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
    def _fill_dictionary(_dict: dict) -> None:
        _fill_num = max([len(val) for _, val in _dict.items()])
        for key, val in _dict.items():
            if len(val) == 0:
                _dict[key] = [np.nan] * _fill_num


RequestedForecast = Tuple[Account, ForecastStrategy, dict]
RequestedForecastList = List[RequestedForecast]


class Forecast:

    def __init__(self, requested_forecast: RequestedForecast):
        self.account: Account = requested_forecast[0]
        self.strategy: ForecastStrategy = requested_forecast[1]
        self.strategy_kwargs: dict = requested_forecast[2]

        self.forecast_data: pd.DataFrame | Dict[str, Forecast] | None = None
        self.forecast_data = self.strategy.predict(account=self.account, **self.strategy_kwargs)


class ForecastFactory:
    @staticmethod
    def __call__(requested_forecast_list: RequestedForecastList,
                 forecasts_dict: Dict[str, Forecast] | None = None) -> Dict[str, Forecast]:
        if forecasts_dict is None:
            forecasts_dict: Dict[str, Forecast] = dict()
        for requested_forecast in requested_forecast_list:
            forcast = Forecast(requested_forecast)
            if isinstance(forcast.forecast_data, pd.DataFrame):
                forecasts_dict[requested_forecast[0].name] = forcast
            elif isinstance(forcast.forecast_data, dict):
                for account_name, forecast_l in forcast.forecast_data.items():
                    forecasts_dict[account_name] = forecast_l
        return forecasts_dict


class ForecastByMeanStrategy(ForecastStrategy):

    def predict(self,
                account: Account,
                predicted_days: int,
                stats: AccountStats | None = None,
                simulation_date: str = "",
                **kwargs) -> pd.DataFrame:

        def _predict(stats: AccountStats | None = None) -> pd.DataFrame:

            if stats is None:
                stats = account.period_stats(date="", last_n_days=365)

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

            self._fill_dictionary(pred_l)
            pred_l = pd.DataFrame(pred_l)

            return pred_l

        force_new = kwargs.get('force_new', False)
        pred: pd.DataFrame = self._prediction_wraper(account=account,
                                                     predict_func=_predict,
                                                     stats=stats,
                                                     simulation_date=simulation_date,
                                                     force_new=force_new)
        account.prediction = pred

        return pred


class NoTransactionsStrategy(ForecastStrategy):

    def predict(self,
                account: Account,
                predicted_days: int,
                stats: AccountStats | None = None,
                balance_offset: float = 0.,
                simulation_date: str = "",
                **kwargs) -> pd.DataFrame | None:

        def _predict() -> pd.DataFrame | None:

            pred_l = {col_name: list() for col_name in account.columns_names}
            pred_l['date'].append(0)
            pred_l['description'].append(PREDICTED_BALANCE)
            pred_l['code'].append("other")
            self._append_amount(account=account, pred=pred_l, amount=balance_offset)

            with tqdm(total=predicted_days,
                      desc=f"{self.__class__.__name__} iterating {account.name}") as pbar:
                for days_ix in range(predicted_days):
                    pred_l['date'].append(days_ix + 1)
                    pred_l['description'].append(PREDICTED_BALANCE)
                    pred_l['code'].append("other")
                    self._append_amount(account=account, pred=pred_l, amount=0)
                    pbar.update(1)

            self._fill_dictionary(pred_l)
            pred_l = pd.DataFrame(pred_l)

            return pred_l

        force_new = kwargs.get('force_new', False)
        balance_offset = kwargs.get('balance_offset', 0.)
        bal_at_sim_date = account.get_balance_at_date(date=simulation_date)

        pred: pd.DataFrame = self._prediction_wraper(account=account,
                                                     predict_func=_predict,
                                                     stats=stats,
                                                     simulation_date=simulation_date,
                                                     force_new=force_new)
        pred.sort_values(by='date', inplace=True)
        pred.reset_index(drop=True, inplace=True)
        pred[account.balance_column_name] = ((pred.loc[:, [account.positive_names[0], account.negative_names[0]]]).sum(axis=1).cumsum() +
                           balance_offset +
                           bal_at_sim_date)

        return pred


class PlannedTransactionsStrategy(ForecastStrategy):

    def predict(self,
                account: Account,
                predicted_days: int,
                stats: AccountStats | None = None,
                balance_offset: float = 0.,
                simulation_date: str = "",
                **kwargs) -> pd.DataFrame | None:

        planned_transations: pd.DataFrame | None = account.get_planned_transactions(start_date=simulation_date,
                                                                                    predicted_days=predicted_days)
        if planned_transations is not None:
            planned_codes = planned_transations.code.unique()
        else:
            planned_codes = list()

        def _predict(stats: AccountStats | None = None,) -> pd.DataFrame | None:

            pred_l = {col_name: list() for col_name in account.columns_names}
            pred_l['date'].append(0)
            pred_l['description'].append(PREDICTED_BALANCE)
            pred_l['code'].append("other")
            self._append_amount(account=account, pred=pred_l, amount=balance_offset)

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

            self._fill_dictionary(pred_l)
            pred_l = pd.DataFrame(pred_l)

            return pred_l

        force_new = kwargs.get('force_new', False)
        balance_offset = kwargs.get('balance_offset', 0.)
        bal_at_sim_date = account.get_balance_at_date(date=simulation_date)

        pred: pd.DataFrame = self._prediction_wraper(account=account,
                                                     predict_func=_predict,
                                                     stats=stats,
                                                     simulation_date=simulation_date,
                                                     force_new=force_new)

        if planned_transations is not None:
            pred = pd.concat([planned_transations, pred], ignore_index=True)

        pred.sort_values(by='date', inplace=True)
        pred.reset_index(drop=True, inplace=True)
        pred[account.balance_column_name] = ((pred.loc[:, [account.positive_names[0], account.negative_names[0]]]).sum(axis=1).cumsum() +
                                             balance_offset +
                                             bal_at_sim_date)

        pred = pred[[*pred.columns[0:-2:].tolist(), pred.columns[-1], pred.columns[-2]]]

        return pred


class MonteCarloStrategy(ForecastStrategy):
    def predict(self,
                account: Account,
                predicted_days: int,
                stats: AccountStats | None = None,
                simulation_date: str = "",
                force_new: bool = False,
                **kwargs) -> pd.DataFrame:

        planned_transactions: pd.DataFrame | None = account.get_planned_transactions(start_date=simulation_date,
                                                                                     predicted_days=predicted_days)
        if planned_transactions is not None:
            planned_codes = list(planned_transactions.code.unique())
        else:
            planned_codes = list()
        planned_codes.extend(['na', 'internal_cashflow', 'credit', 'other'])

        def _predict(stats: AccountStats | None = None,) -> pd.DataFrame | None:

            # TODO: implement parallelization strategy for MonteCarlo iterations

            pred_l: Dict[str, List[Any]] = {col_name: list() for col_name in account.columns_names}
            pred_l[ITERATION] = list()

            balance_offset = kwargs.get("balance_offset", 0)
            mc_iterations = kwargs.get('mc_iterations', 100)

            with tqdm(total=predicted_days * mc_iterations,
                      desc=f"{self.__class__.__name__} iterating {account.name}") as pbar:
                for mc_iteration in range(mc_iterations):
                    pred_l['date'].append(0)
                    pred_l['description'].append(f"{PREDICTED_BALANCE}_reported_balance")
                    pred_l['code'].append("other")
                    pred_l[ITERATION].append(mc_iteration)
                    self._append_amount(account=account, pred=pred_l, amount=balance_offset)
                    for days_ix, _ in enumerate(range(predicted_days)):
                        for unplanned_index in stats[~stats.index.isin(planned_codes)].index:
                            dice = random.random() <= stats.loc[unplanned_index]['daily_prob']
                            if not dice:
                                continue

                            amount = random.gauss(mu=stats.loc[unplanned_index]['mean'],
                                                  sigma=stats.loc[unplanned_index]['std'])
                            amount = amount
                            if amount != 0:
                                pred_l['date'].append(days_ix + 1)
                                pred_l['description'].append(f"{PREDICTED_BALANCE}_{unplanned_index}")
                                pred_l['code'].append("other")
                                pred_l[ITERATION].append(mc_iteration)
                                self._append_amount(account=account, pred=pred_l, amount=amount)
                        pbar.update(1)

                self._fill_dictionary(pred_l)

                pred_l: pd.DataFrame = pd.DataFrame(pred_l)
            return pred_l

        pred = self._prediction_wraper(account=account,
                                       predict_func=_predict,
                                       stats=stats,
                                       simulation_date=simulation_date,
                                       force_new=force_new)

        bal_at_sim_date = account.get_balance_at_date(date=simulation_date)
        if planned_transactions is not None:
            mc_iterations = kwargs.pop('mc_iterations', 0)
            planned_transactions_list = list()
            for i in range(0, mc_iterations):
                planned_transactions[ITERATION] = i
                planned_transactions_list.append(deepcopy(planned_transactions))
            pred = pd.concat([pred, *planned_transactions_list], axis=0)
        pred.sort_values(by=[ITERATION, 'date'], inplace=True)
        pred.reset_index(drop=True, inplace=True)
        cols = [ITERATION, account.positive_names[0], account.negative_names[0]]

        pred[account.balance_column_name] = (pred.loc[:, cols].groupby(by=ITERATION).cumsum().sum(axis=1) +
                                             bal_at_sim_date)
        del cols

        return pred


class FixedLoanPaymentForecastStrategy(ForecastStrategy):

    def predict(self,
                account: Account,
                op_forecast: Forecast,
                loan_account: Account,
                loan_forecast: Forecast,
                loan_rate: float,
                day_of_month: int,
                payment_amount: float | int | None = None) -> Dict[str, Forecast]:

        first_sim_date_op = pd.to_datetime(np.unique(op_forecast.forecast_data.date.array)[0]).date()
        first_sim_date_loan = pd.to_datetime(np.unique(loan_forecast.forecast_data.date.array)[0]).date()

        date_month_pairs = [(pd.to_datetime(date).date().year, pd.to_datetime(date).date().month)
                            for date in np.unique(op_forecast.forecast_data.date.array)]
        date_month_pairs += [(pd.to_datetime(date).date().year, pd.to_datetime(date).date().month)
                             for date in np.unique(loan_forecast.forecast_data.date.array)]

        date_month_pairs = set(date_month_pairs)

        def _apply_date_shift(df: pd.DataFrame) -> pd.DataFrame:
            df['date_diff'] = df.date.diff()
            df['date_diff'] = df['date_diff'].shift(-1)
            df = df.replace(np.nan, datetime.timedelta(days=0))
            df['date_diff'] = df['date_diff'].apply(lambda i: int(i.days))
            return df

        idx = loan_account.transaction_data.date.array.date < first_sim_date_loan

        historical_loan_data: pd.DataFrame = copy(loan_account.transaction_data.loc[idx])
        historical_loan_data = _apply_date_shift(historical_loan_data)

        del idx

        for year, month in sorted(date_month_pairs):

            shifted_loan_pred = copy(loan_forecast.forecast_data)
            shifted_loan_pred = _apply_date_shift(shifted_loan_pred)

            join_view = pd.concat([historical_loan_data, shifted_loan_pred])
            join_view = _apply_date_shift(join_view)
            join_view.sort_values(by='date', inplace=True, ignore_index=True)
            join_view['year'] = join_view.date.apply(lambda d: pd.to_datetime(d).date().year)
            join_view['month'] = join_view.date.apply(lambda d: pd.to_datetime(d).date().month)
            join_view['daily_interest'] = (
                    join_view['date_diff'] * abs(join_view[loan_account.balance_column_name]) * (loan_rate / 365.25)
            )

            grouped_join_view = join_view.loc[:, ["year", "month", "daily_interest"]].groupby(by=["year", "month"]).sum()
            grouped_join_view['daily_interest'] = grouped_join_view['daily_interest'].shift()
            grouped_join_view.replace(np.nan, 0, inplace=True)

            op_internal_transactions = {col: list() for col in op_forecast.forecast_data.columns}
            loan_internal_transactions = {col: list() for col in loan_forecast.forecast_data.columns}

            transaction_date = datetime.date.fromisoformat(f"{year}-{month:02}-{day_of_month:02}")
            idx = pd.IndexSlice

            try:
                grouped_join_view_len = len(grouped_join_view.loc[idx[year, month], :])
                if grouped_join_view_len == 0:
                    continue
                elif grouped_join_view_len == 1:
                    interest_amount = float(grouped_join_view.loc[idx[year, month], 'daily_interest'])
                else:
                    interest_amount = float(grouped_join_view.loc[idx[year, month], 'daily_interest'].values[0])
            except KeyError:
                interest_amount = 0

            del idx

            if first_sim_date_op <= transaction_date:
                for mc_iteration in op_forecast.forecast_data[ITERATION].unique():
                    op_internal_transactions['date'].append(transaction_date)
                    op_internal_transactions['description'].append("predicted interest BNC PR")
                    op_internal_transactions[account.positive_names[0]].append(0)
                    op_internal_transactions[account.negative_names[0]].append(-interest_amount)
                    op_internal_transactions['code'].append('internal_cashflow')
                    op_internal_transactions[ITERATION].append(mc_iteration)
                    op_internal_transactions[account.balance_column_name].append(0)

                    op_internal_transactions['date'].append(transaction_date)
                    op_internal_transactions['description'].append("predicted capital paid BNC PR")
                    op_internal_transactions[account.positive_names[0]].append(0)
                    op_internal_transactions[account.negative_names[0]].append(-(payment_amount - interest_amount))
                    op_internal_transactions['code'].append('internal_cashflow')
                    op_internal_transactions[ITERATION].append(mc_iteration)
                    op_internal_transactions[account.balance_column_name].append(0)

            if first_sim_date_loan <= transaction_date:
                loan_internal_transactions['date'].append(transaction_date)
                loan_internal_transactions['description'].append("predicred interest")
                loan_internal_transactions[account.positive_names[0]].append(interest_amount)
                loan_internal_transactions[account.negative_names[0]].append(-interest_amount)
                loan_internal_transactions['code'].append('internal_cashflow')
                loan_internal_transactions[loan_account.balance_column_name].append(0)

                loan_internal_transactions['date'].append(transaction_date)
                loan_internal_transactions['description'].append("predicted capital paid")
                loan_internal_transactions[account.positive_names[0]].append((payment_amount-interest_amount))
                loan_internal_transactions[account.negative_names[0]].append(0)
                loan_internal_transactions['code'].append('internal_cashflow')
                loan_internal_transactions[loan_account.balance_column_name].append(0)

            self._fill_dictionary(op_internal_transactions)
            op_internal_transactions = pd.DataFrame(op_internal_transactions)
            loan_internal_transactions = pd.DataFrame(loan_internal_transactions)

            op_internal_transactions['date'] = pd.to_datetime(op_internal_transactions['date'])
            loan_internal_transactions['date'] = pd.to_datetime(loan_internal_transactions['date'])

            loan_forecast.forecast_data = pd.concat([loan_forecast.forecast_data, loan_internal_transactions])
            op_forecast.forecast_data = pd.concat([op_forecast.forecast_data, op_internal_transactions])

            loan_forecast.forecast_data.sort_values(by=['date', loan_account.balance_column_name], inplace=True, ignore_index=True)
            op_forecast.forecast_data.sort_values(by=[ITERATION, 'date', account.balance_column_name], inplace=True, ignore_index=True)

            loan_forecast.forecast_data.reset_index(drop=True, inplace=True)
            op_forecast.forecast_data.reset_index(drop=True, inplace=True)

            cols = [loan_account.positive_names[0], loan_account.negative_names[0]]
            loan_forecast.forecast_data[loan_account.balance_column_name] = (
                    loan_forecast.forecast_data.loc[:, cols].cumsum().sum(axis=1) + loan_forecast.forecast_data.loc[0, loan_account.balance_column_name]
            )
            cols = [ITERATION, account.positive_names[0], account.negative_names[0]]
            op_forecast.forecast_data[account.balance_column_name] = (
                    (op_forecast.forecast_data.loc[:, cols].groupby(by=ITERATION).cumsum().sum(axis=1)) + op_forecast.forecast_data.loc[0, account.balance_column_name]
            )

        return {account.name: op_forecast,
                loan_account.name: loan_forecast}


class CreditCardPaymentForecastStrategy(ForecastStrategy):

    @staticmethod
    def get_balance_at_date(df: pd.DataFrame, date: str | datetime.date):
        if not isinstance(date, datetime.date):
            date = datetime.date.fromisoformat(date)

        are_near = abs(df.index.array.date - date) == min(abs(df.index.array.date - date))
        if df.index.array.date[0] > date:
            bal = 0
            for ix, near_idx in enumerate(are_near):
                if near_idx:
                    if ix > 0:
                        bal = df.iloc[ix-1, 0]
                        break
        else:
            bal = df[are_near].iloc[-1:, 0].item()
        return bal

    def predict(self,
                account: Account,
                op_forecast: Forecast,
                cc_account: Account,
                cc_forecast: Forecast) -> Dict[str, Forecast]:

        """
        :param op_forecast:         Prediction of the account from which the amount is taken.
        :param cc_forecast:           Prediction of the account where the amount is deposited.
        :param day_of_month:        The date at which the internal transaction will be done.
        :param payment_amount:              The amount off the transaction. Optional if amount_at_date is specified.
        :param amount_at_date:      The amount at a previous date which will be determined at runtime.
                                    Optional, has no effect if 'amount' is specified. At leas one of the two must
                                    be specified.
        :return:                    A prediction for both accounts reflecting the internal monthly transaction and
                                    the interest if applicable.
        """

        first_sim_date_op = pd.to_datetime(np.unique(op_forecast.forecast_data.date.array)[0]).date()
        last_sim_date_op = pd.to_datetime(np.unique(op_forecast.forecast_data.date.array)[-1]).date()

        first_sim_date_loan = pd.to_datetime(np.unique(cc_forecast.forecast_data.date.array)[0]).date()
        last_sim_date_loan = pd.to_datetime(np.unique(cc_forecast.forecast_data.date.array)[-1]).date()

        date_month_pairs = [(pd.to_datetime(date).date().year, pd.to_datetime(date).date().month)
                            for date in np.unique(op_forecast.forecast_data.date.array)]
        date_month_pairs += [(pd.to_datetime(date).date().year, pd.to_datetime(date).date().month)
                             for date in np.unique(cc_forecast.forecast_data.date.array)]

        date_month_pairs = set(date_month_pairs)

        idx = cc_account.transaction_data.date.array.date < first_sim_date_loan

        historical_cc_data: pd.DataFrame = copy(cc_account.transaction_data.loc[idx])
        historical_cc_data[cc_account.balance_column_name] = cc_account.get_balance()

        del idx

        for year, month in sorted(date_month_pairs):

            shifted_cc_pred = copy(cc_forecast.forecast_data)
            join_view = pd.concat([historical_cc_data, shifted_cc_pred])

            op_pred_l = {col: list() for col in op_forecast.forecast_data.columns}
            cc_pred_l = {col: list() for col in cc_forecast.forecast_data.columns}

            transaction_date = datetime.date.fromisoformat(f"{year}-{month:02}-{cc_account.statement_day:02}")

            cols = ['date', cc_account.balance_column_name]
            statement_amount = self.get_balance_at_date(df=join_view.loc[:, cols].groupby(by='date').mean(),
                                                        date=transaction_date)
            del cols

            if first_sim_date_op <= transaction_date <= last_sim_date_op:
                for mc_iteration in op_forecast.forecast_data[ITERATION].unique():
                    op_pred_l['date'].append(transaction_date)
                    op_pred_l['description'].append("predicted interest BNC PR")
                    op_pred_l[account.positive_names[0]].append(0)
                    op_pred_l[account.negative_names[0]].append(statement_amount)
                    op_pred_l['code'].append('internal_cashflow')
                    op_pred_l[ITERATION].append(mc_iteration)
                    op_pred_l[account.balance_column_name].append(0)

            if first_sim_date_loan <= transaction_date <= last_sim_date_loan:
                for mc_iteration in cc_forecast.forecast_data[ITERATION].unique():
                    cc_pred_l['date'].append(transaction_date)
                    cc_pred_l['description'].append("predicted cc payment")
                    cc_pred_l[cc_account.positive_names[0]].append(-statement_amount)
                    cc_pred_l[cc_account.negative_names[0]].append(0)
                    cc_pred_l['code'].append('internal_cashflow')
                    cc_pred_l[ITERATION].append(mc_iteration)
                    cc_pred_l[cc_account.balance_column_name].append(0)

            self._fill_dictionary(op_pred_l)
            self._fill_dictionary(cc_pred_l)

            op_pred_l = pd.DataFrame(op_pred_l)
            cc_pred_l = pd.DataFrame(cc_pred_l)

            op_pred_l['date'] = pd.to_datetime(op_pred_l['date'])
            cc_pred_l['date'] = pd.to_datetime(cc_pred_l['date'])

            cc_forecast.forecast_data = pd.concat([cc_forecast.forecast_data, cc_pred_l])
            op_forecast.forecast_data = pd.concat([op_forecast.forecast_data, op_pred_l])

            cc_forecast.forecast_data.sort_values(by=[ITERATION, 'date', cc_account.balance_column_name],
                                                  inplace=True, ignore_index=True)
            op_forecast.forecast_data.sort_values(by=[ITERATION, 'date', account.balance_column_name],
                                                  inplace=True, ignore_index=True)

            cc_forecast.forecast_data.reset_index(drop=True, inplace=True)
            op_forecast.forecast_data.reset_index(drop=True, inplace=True)

            cols = [ITERATION, cc_account.positive_names[0], cc_account.negative_names[0]]
            cc_forecast.forecast_data[cc_account.balance_column_name] = cc_forecast.forecast_data.loc[:, cols].groupby(by=ITERATION).cumsum().sum(axis=1) + cc_forecast.forecast_data.loc[0, cc_account.balance_column_name]
            cols = [ITERATION, account.positive_names[0], account.negative_names[0]]
            op_forecast.forecast_data[account.balance_column_name] = op_forecast.forecast_data.loc[:, cols].groupby(by=ITERATION).cumsum().sum(axis=1) + op_forecast.forecast_data.loc[0, account.balance_column_name]

            del cols

        return {account.name: op_forecast,
                cc_account.name: cc_forecast}
