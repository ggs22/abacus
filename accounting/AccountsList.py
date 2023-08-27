from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accounting.Account import Account, PREDICTED_BALANCE


class AccountsList:

    def __init__(self, accounts: List[Account]):
        self.acounts = accounts

    def __getitem__(self, item) -> Account:
        return self.acounts[item]

    def get_balance_prediction(self, predicted_days: int = 365) -> pd.DataFrame:
        preds: List[pd.DataFrame] = list()
        for account in self:
            if account.status == "OPEN":
                pred: pd.DataFrame = account.get_balance_prediction(predicted_days=predicted_days)
                pred = pred.loc[:, ['date', 'description', 'balance']].groupby(by='date').last()
                for swap_col in ['balance', 'description']:
                    pred[f'{swap_col}_{account.name}'] = pred[swap_col]
                    pred.drop(columns=swap_col, inplace=True)
                pred = pred.groupby(by='date').last()
                preds.append(pred)
        pred = pd.concat(preds, axis=1).ffill(axis=0).replace(np.nan, 0)
        pred['total_balance'] = pred.sum(axis=1, numeric_only=True)
        return pred

    def plot_balance_prediction(self, predicted_days: int = 365) -> None:
        pred = self.get_balance_prediction(predicted_days)
        for col in pred.columns:
            if 'description' not in col:
                plt.plot(pred.index, pred[col])
            else:
                ix = pred[col] == PREDICTED_BALANCE
                plt.vlines(x=pred[ix].index[0], ymin=-7e4, ymax=2e4, linestyles='--')
        plt.grid(visible=True)
        plt.legend(pred.columns[pred.columns != 'description'])
        plt.title(f"Prediction all open acounts")
