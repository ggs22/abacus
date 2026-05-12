from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go

from accounting.Account import Account
from accounting.forecast_strategies import Forecast

if TYPE_CHECKING:
    from accounting.account_list import AccountsList


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a hex color string to an rgba() CSS string."""
    if color.startswith('#'):
        r, g, b = plotly.colors.hex_to_rgb(color)
        return f'rgba({r},{g},{b},{alpha})'
    return color


def make_balance_trace(account: Account, c: str | None = None) -> go.Scatter:
    """Return a single Scatter trace for an account's balance over time."""
    bal = account.balance_column
    return go.Scatter(
        x=bal.index,
        y=bal.iloc[:, 0],
        name=account.name,
        mode='lines',
        line=dict(color=c or account.color)
    )


def plot_account(account: Account, c: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(make_balance_trace(account, c))
    fig.update_layout(title=account.name, xaxis_title='Date', yaxis_title='Balance',
                      showlegend=True)
    return fig


def histplot_account(account: Account, start_date: str, end_date: str = "",
                     c: str | None = None) -> go.Figure:
    data, _ = account.get_period_data(start_date=start_date, end_date=end_date)
    title = f"Histplot {account.name} - {start_date}" + (f" {end_date}" if end_date else "")
    fig = go.Figure()
    if data is not None:
        data = data.copy()
        cols = [col for col, _ in account.conf.numerical_columns]
        signs = [sign for _, sign in account.conf.numerical_columns]
        data['merged'] = (data.loc[:, cols] * signs).sum(axis=1)
        fig.add_trace(go.Histogram(
            x=data['merged'],
            name=account.name,
            marker_color=c or account.color
        ))
    fig.update_layout(title=title, xaxis_title='Amount', yaxis_title='Count')
    return fig


def barplot_account(account: Account, start_date: str, end_date: str = "") -> go.Figure:
    data, period_length = account.get_period_data(start_date=start_date, end_date=end_date)
    fig = go.Figure()
    if data is not None:
        data = data.groupby(by='code').sum(numeric_only=True)
        data['total'] = (data.loc[:, account.numerical_names] * account.numerical_signs).sum(axis=1)
        data.sort_values(by='total', ascending=False, inplace=True)

        income = data[data['total'] > 0]['total'].sum()
        expenses = data[data['total'] <= 0]['total'].sum()
        colors = ['#4a9e5a' if v > 0 else '#b05050' for v in data['total']]
        text = [f"{v:.2f} / {v / period_length:.2f}" for v in data['total']]

        fig.add_trace(go.Bar(
            x=data['total'],
            y=data.index,
            orientation='h',
            marker_color=colors,
            text=text,
            textposition='outside',
            name=account.name
        ))
        fig.update_layout(
            title=(f"{account.name} - {start_date}<br>"
                   f"in: {income:.2f}, out: {expenses:.2f}, bal: {income + expenses:.2f}"),
            xaxis_title='Amount',
            yaxis_title='Category'
        )
    return fig


def plot_forecast(forecast: Forecast, c: str = 'blue',
                  fig: go.Figure | None = None) -> go.Figure:
    """
    Add a mean line + ±std band for one forecast.

    If fig is provided, traces are added to it (used by AccountsList.plot_forecasts
    to compose multiple forecasts into a single figure). Otherwise a new figure is
    created and layout is applied.
    """
    is_new = fig is None
    if is_new:
        fig = go.Figure()

    bal_col = forecast.account.balance_column_name
    grouped = forecast.forecast_data.loc[:, ['date', bal_col]].groupby(by='date')
    mean = grouped.mean()
    std = grouped.std().fillna(0)

    fill_color = hex_to_rgba(c, 0.3)

    # Upper envelope — invisible, acts as ceiling for tonexty fill
    fig.add_trace(go.Scatter(
        x=mean.index, y=(mean + std)[bal_col],
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip', name=''
    ))
    # Lower envelope — fills back to upper trace
    fig.add_trace(go.Scatter(
        x=mean.index, y=(mean - std)[bal_col],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor=fill_color,
        showlegend=False, hoverinfo='skip', name=''
    ))
    # Mean line
    fig.add_trace(go.Scatter(
        x=mean.index, y=mean[bal_col],
        mode='lines', line=dict(dash='dash', color=c),
        name=forecast.account.name
    ))

    if is_new:
        fig.update_layout(title=forecast.account.name, xaxis_title='Date',
                          yaxis_title='Balance', showlegend=True)
    return fig


def plot_accounts_list(accounts_list: "AccountsList", title: str = "Accounts", show_total: bool = True) -> go.Figure:
    fig = go.Figure()
    for account in accounts_list:
        fig.add_trace(make_balance_trace(account))
    if show_total:
        total = accounts_list.balance_column
        fig.add_trace(go.Scatter(
            x=total.index, y=total['balance_total'],
            name='Total', mode='lines',
            line=dict(color=accounts_list.color, dash='dash')
        ))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Balance',
                      showlegend=True)
    return fig


def plot_cumulative_balances(accounts_list: "AccountsList", accounts: List[Account],
                             title: str = "") -> go.Figure:
    total = accounts_list.get_balance_columns_sum(accounts)
    label = "summed balance: " + ", ".join(acc.name for acc in accounts)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=total.index, y=total['balance_total'],
        name=label, mode='lines',
        line=dict(color=accounts_list.color)
    ))
    fig.update_layout(title=title or "Cumulative Balance", xaxis_title='Date',
                      yaxis_title='Balance', showlegend=True)
    return fig


def plot_forecasts(accounts_list: "AccountsList",
                   forecasts: Dict[str, Forecast],
                   total_offset: float = 0.,
                   show_history: bool = True,
                   sum_accounts: list[str] | None = None,
                   title: str = "") -> go.Figure:
    fig = go.Figure()
    mean_bal = list()
    std_bal = list()

    sum_set = set(sum_accounts) if sum_accounts else set()

    for account_name, forecast in forecasts.items():
        c = accounts_list.accounts[account_name].color
        if show_history:
            bal = forecast.account.balance_column
            fig.add_trace(go.Scatter(
                x=bal.index, y=bal.iloc[:, 0],
                name=forecast.account.name,
                mode='lines', line=dict(color=c),
                legendgroup=account_name,
                showlegend=False,
            ))
        plot_forecast(forecast, c=c, fig=fig)

        if account_name in sum_set:
            bal_col = forecast.account.balance_column_name
            mean_bal.append(
                forecast.forecast_data.loc[:, ['date', bal_col]].groupby(by='date').mean()
            )
            std_bal.append(
                forecast.forecast_data.loc[:, ['date', bal_col]].groupby(by='date').std()
            )

    if sum_set:
        # Historical sum
        hist_balances = [
            accounts_list.accounts[n].balance_column.groupby(level=0).last().ffill()
            for n in sum_set if n in accounts_list.accounts
        ]
        if hist_balances:
            c = accounts_list.color
            hist_total = pd.concat(hist_balances, axis=1).ffill().sum(axis=1)
            fig.add_trace(go.Scatter(
                x=hist_total.index, y=hist_total.values,
                name="Sum", mode='lines',
                line=dict(color=c),
                showlegend=False,
            ))

    if mean_bal:
        total = pd.concat(mean_bal, axis=1).ffill().dropna()
        total_std = pd.concat(std_bal, axis=1).ffill().dropna()
        total['balance_total'] = total.sum(axis=1) + total_offset
        total_std['balance_std'] = total_std.std(axis=1)

        c = accounts_list.color
        fill_color = hex_to_rgba(c, 0.3)

        fig.add_trace(go.Scatter(
            x=total.index,
            y=(total['balance_total'] + total_std['balance_std']).values,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip', name=''
        ))
        fig.add_trace(go.Scatter(
            x=total.index,
            y=(total['balance_total'] - total_std['balance_std']).values,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=fill_color,
            showlegend=False, hoverinfo='skip', name=''
        ))
        fig.add_trace(go.Scatter(
            x=total.index, y=total['balance_total'].values,
            mode='lines', line=dict(dash='dash', color=c),
            name='Total'
        ))

    fig.update_layout(title=title or "Forecasts", xaxis_title='Date',
                      yaxis_title='Balance', showlegend=True)
    return fig


def barplot(accounts_list: "AccountsList",
            start_date: str,
            end_date: str = "",
            excluded_codes: Sequence[str] = ('internal_cashflow', 'credit', 'expense_account'),
            title: str = "") -> go.Figure:
    data_df = list()
    overall_len = 0
    for acc in accounts_list:
        data, period_length = acc.get_period_data(start_date=start_date, end_date=end_date)
        if data is not None:
            overall_len = max(overall_len, period_length)
            data = data.groupby(by='code').sum(numeric_only=True)
            data['total'] = (data.loc[:, acc.numerical_names] * acc.numerical_signs).sum(axis=1)
            data.sort_values(by='total', ascending=False, inplace=True)
            data_df.append(data)

    fig = go.Figure()
    if data_df:
        data = pd.concat(data_df, axis=1).loc[:, ['total']].sum(axis=1)
        data.sort_values(inplace=True, ascending=False)

        for code in excluded_codes:
            if code in data.index:
                data.drop(labels=code, inplace=True)

        income = data[data > 0].sum()
        expenses = data[data <= 0].sum()
        colors = ['#4a9e5a' if v > 0 else '#b05050' for v in data]
        text = [f"{v:.2f} / {v / overall_len:.2f}" for v in data]

        chart_title = (
            (title or f"Accounts barplot - {start_date}") +
            f"<br>in: {income:.2f}, out: {expenses:.2f}, bal: {income + expenses:.2f}"
        )

        fig.add_trace(go.Bar(
            x=data.values,
            y=data.index,
            orientation='h',
            marker_color=colors,
            text=text,
            textposition='outside',
            name=start_date
        ))
        fig.update_layout(title=chart_title, xaxis_title='Amount',
                          yaxis=dict(title='Category', automargin=True),
                          margin=dict(r=160),
                          showlegend=False)
    return fig
