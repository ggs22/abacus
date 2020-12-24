import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from CSV_Parser import CSV_Parser
from pathlib import Path


def set_plot_fore_color(fig: plt.Figure, plot_area_color=(0.35, 0.35, 0.35), fore_color=(0.85, 0.85, 0.85)):
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


def plot_cashflow(data: pd.DataFrame, x_col: str, y_col: str, legend=None, x_label='', y_label='', color='g'):
    fig = plt.figure()
    data = data.sort_values(by='date')

    set_plot_fore_color(fig=fig)

    plt.plot(data.loc[:, x_col], data.loc[:, y_col], color=color)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation=90)
    if legend is not None:
        plt.legend(legend)
    plt.pause(0.05)
    plt.show()


def barplot_total(data_desjardins=None, data_co=None):

    _data = None
    if data_desjardins is not None:
        _data = data_desjardins.copy()
        _data['debit'] = _data['withdrawal']
        _data['credit'] = _data['deposit']
        del _data['withdrawal']
        del _data['deposit']

        _data['debit'] = -_data['debit']
        _data = _data[_data['code'] != 'internal_cashflow']
    if data_co is not None:
        if _data is not None:
            _tmp = data_co.copy()
            _tmp['debit'] = -_tmp['debit']
            _data = pd.concat([_tmp, _data], axis=0)
            _data = _data[_data['code'] != 'internal_cashflow']
        else:
            _data = data_co.copy()
            _data['debit'] = -_data['debit']
            _data = _data[_data['code'] != 'internal_cashflow']

    fig = plt.figure('')

    _df = _data.loc[:, ['credit', 'debit', 'code']].groupby(by='code').sum()
    abs_max = np.max([_df['debit'].abs().max(), _df['credit'].abs().max()])
    abs_max = int(abs_max * 1.1)

    set_plot_fore_color(fig=fig)

    sns.barplot(data=_data, x='code', y='debit',
                estimator=sum, color='r', errwidth=0)
    sns.barplot(data=_data, x='code', y='credit',
                estimator=sum, color='g', errwidth=0)

    plt.ylim([-abs_max, abs_max])

    plt.xticks(rotation=90)
    plt.savefig(f'images/total.png')
    plt.show()

def barplot_desjardins(data: pd.DataFrame, fig_name):
    fig = plt.figure('Barplot Desjardins')

    set_plot_fore_color(fig=fig)

    data['withdrawal'] = -data['withdrawal']
    data = data[data['code'] != 'internal_cashflow']
    sns.barplot(data=data, x='code', y='withdrawal',
                estimator=sum, color='r', errwidth=0)
    sns.barplot(data=data, x='code', y='deposit',
                estimator=sum, color='g', errwidth=0)
    plt.xticks(rotation=90)
    plt.savefig(f'images/{fig_name}.png')
    plt.show()


def plot_capital_one_debit(data: pd.DataFrame, fig_name):
    fig = plt.figure(f'{fig_name}')

    set_plot_fore_color(fig=fig)

    data['debit'] = -data['debit']
    data = data[data['code'] != 'internal_cashflow']
    sns.barplot(data=data, x='code', y='debit',
                estimator=sum, color='r', errwidth=0)
    plt.xticks(rotation=90)

    plt.savefig(f'images/{fig_name}.png')
    plt.show()


class Grapher:
    def __init__(self, parser: CSV_Parser):
        plt.ion()
        self._parser = parser
        self._operations_account_data, self._credit_line_data, self._student_loan_data, self._capital_one_data \
            = self._parser.get_data()
        Path.mkdir(Path('images'), exist_ok=True)

    # def plot

    def plot_capital_one(self):
        plot_cashflow(self._capital_one_data, 'date', 'debit')

    def plot_desjardins_mc(self):
        plot_cashflow(self._credit_line_data, 'date', 'balance', x_label='date', y_label='balance')

    def plot_desjardins_op(self):
        plot_cashflow(self._operations_account_data, 'date', 'balance')

    def plot_year_total(self, year: int):
        year_op, _, _, year_capital_one = self._parser.get_data_by_date(year=year)
        if year_op.shape[0] > 0:
            data_desjardins = year_op.copy()
        else:
            data_desjardins = None
        if year_capital_one.shape[0] > 0:
            data_co = year_capital_one.copy()
        else:
            data_co = None
        barplot_total(data_desjardins=data_desjardins, data_co=data_co)

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