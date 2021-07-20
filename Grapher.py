import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from CSVParser import CSVParser
from pathlib import Path
from io import BytesIO


def _set_plot_fore_color(fig: plt.Figure, plot_area_color=(0.35, 0.35, 0.35), fore_color=(0.85, 0.85, 0.85)):
    fig.set_facecolor(plot_area_color)
    ax = plt.gca()
    ax.grid(linestyle='--', zorder=0.0)
    ax.set_facecolor(plot_area_color)
    ax.spines['bottom'].set_color(fore_color)
    ax.spines['top'].set_color(fore_color)
    ax.spines['right'].set_color(fore_color)
    ax.spines['left'].set_color(fore_color)
    ax.tick_params(axis='x', colors=fore_color)
    ax.tick_params(axis='y', colors=fore_color)
    ax.xaxis.label.set_color(fore_color)
    ax.yaxis.label.set_color(fore_color)


def _plot_cashflow(data: pd.DataFrame, x_col: str, y_col: str, legend=None, x_label='', y_label='', color='g'):
    fig = plt.figure(figsize=(5, 5))
    data = data.sort_values(by='date')

    _set_plot_fore_color(fig=fig)

    plt.plot(data.loc[:, x_col], data.loc[:, y_col], color=color)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation=90)
    if legend is not None:
        plt.legend(legend)
    plt.pause(0.05)
    # plt.show()
    return fig


def barplot_desjardins(data: pd.DataFrame, fig_name: str = None, buffer=None, transaction: str = 'withdrawal',
                       fig_size: tuple = None):
    '''
    :param data: Dataframe containing the Desjardins bank transactions information
    :param fig_name: Optionnal figure name that will cause the figure to be saved
    :param buffer: Optionnal memory buffer to output the figure
    :param transaction: str, 'withdrawal' or 'deposit'
    :param fig_size: Pyplot figure size, also determines Tkinter display size
    :return: The pyplot figure of the plot
    '''

    fig = plt.figure('Barplot Desjardins', figsize=fig_size)

    _set_plot_fore_color(fig=fig)

    # internal cashflows are not interesting
    _data = data[data['code'] != 'internal_cashflow']
    # Lets sort the relevant data out
    _data = _data.loc[:, ['code', 'withdrawal']].groupby(by='code').sum()
    _data = _data.sort_values(by='withdrawal', ascending=False)
    # Display the data of interest
    sns.barplot(data=_data, x=transaction, y=_data.index, color='r', errwidth=0)

    plt.yticks([])
    plt.ylabel('')

    for i, ix in enumerate(_data.index):
        plt.text(x=_data.loc[ix, 'withdrawal'], y=i, s=ix)

    if fig_name is not None:
        plt.savefig(f'images/{fig_name}.png')

    if buffer is not None:
        plt.savefig(buffer)
        buffer.seek(0)
    else:
        plt.close()

    return fig


def plot_capital_one_debit(data: pd.DataFrame, fig_name):
    fig = plt.figure(f'{fig_name}')

    _set_plot_fore_color(fig=fig)

    data['debit'] = -data['debit']
    data = data[data['code'] != 'internal_cashflow']
    sns.barplot(data=data, x='code', y='debit',
                estimator=sum, color='r', errwidth=0)
    plt.xticks(rotation=90)

    plt.savefig(f'images/{fig_name}.png')
    plt.show()


class Grapher:
    def __init__(self, accounts: dict):
        plt.ion()

        self._operations_account_data = accounts['desjardins_op']
        self._credit_line_data = accounts['desjardins_mc']
        self._student_loan_data = accounts['desjardins_sl']
        self._ppcard_data = accounts['desjardins_pp']
        self._capital_one_data = accounts['capital_one']

        Path.mkdir(Path('images'), exist_ok=True)

    # def plot

    def plot_capital_one(self):
        return _plot_cashflow(self._capital_one_data, 'date', 'debit')

    def plot_desjardins_mc(self):
        return _plot_cashflow(self._credit_line_data, 'date', 'balance', x_label='date', y_label='balance')

    def plot_desjardins_op(self):
        return _plot_cashflow(self._operations_account_data, 'date', 'balance')

    def plot_year_total(self, year: int):

        _data = self._parser.get_combine_op_and_co()
        _data[_data['date'].array.year == year]
        _data['debit'] *= -1
        fig = plt.figure('')

        _df = _data.loc[:, ['credit', 'debit', 'code']].groupby(by='code').sum()
        abs_max = np.max([_df['debit'].abs().max(), _df['credit'].abs().max()])
        abs_max = int(abs_max * 1.1)

        _set_plot_fore_color(fig=fig)

        sns.barplot(data=_data, x='code', y='debit',
                    estimator=sum, color='r', errwidth=0)
        sns.barplot(data=_data, x='code', y='credit',
                    estimator=sum, color='g', errwidth=0)

        plt.ylim([-abs_max, abs_max])

        plt.xticks(rotation=90)
        plt.savefig(f'images/total.png')
        plt.show()

    def plot_year_desjardins(self, year: int):
        year_op, year_cl, year_sl, _ = self._parser.get_data_by_date(year=year)
        if year_op.shape[0] >= 0:
            data = year_op.copy()
            barplot_desjardins(data=data, fig_name=f'Capital_one_expenses_{year}')

    def plot_year_capital_one(self, year: int):
        _, _, _, year_capital_one = self._parser.get_data_by_date(year=year)
        if year_capital_one.shape[0] == 0:
            return
        data = year_capital_one.copy()

        plot_capital_one_debit(data=data, fig_name=f'Capital One Expenses {year}')

    def plot_month_capital_one(self, month: int):
        _, _, _, month_capital_one = self._parser.get_data_by_date(month=month, year=2020)
        if month_capital_one.shape[0] == 0:
            return
        data = month_capital_one.copy()
        plot_capital_one_debit(data=data, fig_name=f'co_m{month}')

    def plot_all_months_capital_one(self):
        for month in range(1, 13):
            self.plot_month_capital_one(month=month)

    def plot_total(self):
        _data = self._parser.get_combine_op_and_co()
        _data['debit'] *= -1

        fig = plt.figure('')

        _df = _data.loc[:, ['credit', 'debit', 'code']].groupby(by='code').sum()
        abs_max = np.max([_df['debit'].abs().max(), _df['credit'].abs().max()])
        abs_max = int(abs_max * 1.1)

        _set_plot_fore_color(fig=fig)

        sns.barplot(data=_data, x='code', y='debit',
                    estimator=sum, color='r', errwidth=0)
        sns.barplot(data=_data, x='code', y='credit',
                    estimator=sum, color='g', errwidth=0)

        plt.ylim([-abs_max, abs_max])

        plt.xticks(rotation=90)
        plt.savefig(f'images/total.png')
        plt.show()
