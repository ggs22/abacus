from typing import List

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from accounting.Account import Account
from accounting.prediction_strategies import PredictionStrategy


def finalize_plot(plot_func: callable):
    def plot_func_wraper(self, *args, **kwargs):
        plot_func(self, *args, **kwargs)
        plt.grid(visible=True)
        plt.legend()

    return plot_func_wraper


class AccountsList:

    def __init__(self, accounts: List[Account]):
        self.accounts = dict()
        self.__acclist__ = list()
        for account in accounts:
            self.accounts[account.name] = account
            self.__acclist__.append(account)

        self.color = cm.get_cmap('tab10')(5)

    def __getitem__(self, ix) -> Account:
        if isinstance(ix, int):
            item = self.__acclist__[ix]
        elif isinstance(ix, str):
            item = self.accounts[ix]
        return item

    @property
    def balance_columns(self) -> pd.DataFrame:
        balances = list()
        for account in self:
            balance = account.balance_column
            balance = balance.groupby(level=0).last()
            balances.append(balance)

        total = pd.concat(balances, axis=1)

        for col in total.columns:
            acc = self[col.split('_')[1]]
            ix = total.index < acc.most_recent_date.item()
            total.loc[ix, col] = total.loc[ix, col].ffill()

            if acc.status == "OPEN":
                total[col].ffill(inplace=True)

        total['balance_total'] = total.sum(axis=1)

        return total

    def get_balance_prediction(self, predicted_days: int = 365, **kwargs) -> pd.DataFrame:
        preds: List[pd.DataFrame] = list()
        for account in self:
            pred: pd.DataFrame = self.predict_strategy.predict(account=account,
                                                               predicted_days=predicted_days,
                                                               **kwargs)
            if pred is not None:
                reduced_pred = pred.loc[:, ['date', 'description', 'balance']].groupby(
                    by=['description', 'date'], group_keys=False
                ).last()
                reduced_pred[f'balance_{account.name}'] = reduced_pred['balance']
                reduced_pred.drop(columns='balance', inplace=True)
                preds.append(reduced_pred)
        pred = pd.concat(preds, axis=1).sort_index(axis=0, level=[0, 1])
        pred.ffill(inplace=True)
        pred['total_balance'] = pred.sum(axis=1, numeric_only=True)
        pred.dropna(inplace=True)
        return pred

    @finalize_plot
    def plot(self, fig_name: str = "") -> None:

        if fig_name == "":
            fig_name = "Accounts plot"

        for account in self:
            account.plot(figure_name=fig_name)
        total = self.balance_columns

        plt.plot(total['balance_total'], label="total balance", c=self.color)

    @finalize_plot
    def plot_predictions(self,
                         predict_strategy: PredictionStrategy,
                         predicted_days: int = 356,
                         figure_name: str = "",
                         simulation_date: str = "",
                         **kwargs):
        mean_bal = list()
        std_bal = list()
        for acc in self:
            predict_strategy.plot_prediction(account=acc,
                                             predicted_days=predicted_days,
                                             figure_name=figure_name,
                                             simulation_date=simulation_date,
                                             **kwargs)
            if acc.prediction is not None:
                pred = acc.prediction.loc[:, ['date', 'balance']].rename(columns={'balance': f'balance_{acc.name}'})
                mean_bal.append(pred.groupby(by='date').mean())
                std_bal.append(pred.groupby(by='date').std())
        total = pd.concat(mean_bal, axis=1)
        total = total.ffill().dropna()
        total_std = pd.concat(std_bal, axis=1)
        total_std = total_std.ffill().dropna()
        total['balance_total'] = total.sum(axis=1)
        total_std['balance_std'] = total_std.std(axis=1)
        plt.plot(total['balance_total'], linestyle="--", label="", c=self.color)
        plt.fill_between(x=total.index,
                         y1=total['balance_total']-total_std['balance_std'],
                         y2=total['balance_total']+total_std['balance_std'],
                         color=self.color,
                         alpha=0.5)

    def barplot(self, period_seed_date: str, date_end: str = ""):
        data_df = list()
        overall_len = 0
        for acc in self:
            data, period_length = acc.get_period_data(period_seed_date=period_seed_date, date_end=date_end)
            if data is not None:
                overall_len = max(overall_len, period_length)
                data: pd.DataFrame = data.groupby(by='code').sum(numeric_only=True)
                data['total'] = (data.loc[:, acc.numerical_names] * acc.numerical_signs).sum(axis=1)
                data.sort_values(by='total', ascending=False, inplace=True)

                data_df.append(data)

        data = pd.concat(data_df, axis=1).loc[:, ['total']].sum(axis=1)
        data.sort_values(inplace=True, ascending=False)

        income = data[data > 0].sum()
        expenses = data[data <= 0].sum()

        plt.figure(num=f"All accounts barplot - {period_seed_date}")
        plt.title(label=f"All accounts barplot - {period_seed_date}\n"
                        f"in: {income: .2f}, out: {expenses: .2f}, bal: {income + expenses: .2f}")

        colors = list()
        for amount in data:
            color = (.75, 0, 0) if amount <= 0 else (0, .75, 0)
            colors.append(color)
        plt.barh(y=data.index, width=data, color=colors)
        for stat in data.index:
            plt.text(x=max(0, data[stat]) + 10,
                     y=stat,
                     s=f"{data[stat]: .2f} / {data[stat] / overall_len: .2f}")