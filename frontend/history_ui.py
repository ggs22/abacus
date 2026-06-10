import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

from backend.account_list import AccountsList
from backend.plotting import make_balance_trace
from .i18n import t

_DARK_LAYOUT = dict(paper_bgcolor="#1e1e2e", plot_bgcolor="#181825", font_color="#cdd6f4")
_LIGHT_LAYOUT = dict(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")


def _apply_theme(fig: go.Figure, theme: str) -> go.Figure:
    if theme == "dark":
        fig.update_layout(template="plotly_dark", **_DARK_LAYOUT)
    else:
        fig.update_layout(template="plotly_white", **_LIGHT_LAYOUT)
    return fig


def history_layout(account_options: list, default_accounts: list, lang: str = "en") -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(t(lang, "individual_accounts"),
                                       style={"fontWeight": "bold", "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="history-individual-selector",
                                options=account_options,
                                value=default_accounts,
                                multi=True,
                                placeholder=t(lang, "ph_select_accounts"),
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label(t(lang, "sum_group"),
                                       style={"fontWeight": "bold", "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="history-sum-selector",
                                options=account_options,
                                value=[],
                                multi=True,
                                placeholder=t(lang, "ph_select_sum"),
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "marginBottom": "8px"},
            ),
            dcc.Graph(id="history-chart", style={"height": "70vh"}),
        ],
        style={"padding": "10px"},
    )


def register_history_callbacks(app, all_accounts: AccountsList) -> None:

    @app.callback(
        Output("account-selector", "options"),
        Output("account-selector", "value"),
        Output("history-individual-selector", "options"),
        Output("history-individual-selector", "value"),
        Output("history-sum-selector", "options"),
        Input("filtered-accounts", "data"),
        State("account-selector", "value"),
        State("history-individual-selector", "value"),
    )
    def update_history_selector(filtered_names, current_selection, current_individual):
        options = [{"label": n, "value": n} for n in filtered_names]
        return options, filtered_names, options, filtered_names, options

    @app.callback(
        Output("history-chart", "figure"),
        Input("history-individual-selector", "value"),
        Input("history-sum-selector", "value"),
        Input("global-period", "data"),
        State("lang", "data"),
        State("theme", "data"),
    )
    def update_history(individual_accounts, sum_accounts, period_data, lang, theme):
        lang = lang or "en"
        theme = theme or "light"
        fig = go.Figure()
        if individual_accounts:
            individual = AccountsList([all_accounts[n] for n in individual_accounts])
            for account in individual:
                fig.add_trace(make_balance_trace(account))
        if sum_accounts:
            sum_list = AccountsList([all_accounts[n] for n in sum_accounts])
            total = sum_list.balance_column
            label = t(lang, "sum_group") + ":<br>" + "<br>".join(sum_accounts)
            fig.add_trace(go.Scatter(
                x=total.index, y=total["balance_total"],
                name=label, mode="lines",
                line=dict(color=sum_list.color, dash="dash"),
            ))
        layout_kwargs = dict(
            title=t(lang, "chart_history_title"),
            xaxis_title=t(lang, "axis_date"),
            yaxis_title=t(lang, "axis_balance"),
            showlegend=True,
        )
        start = (period_data or {}).get("start")
        end = (period_data or {}).get("end")
        if start or end:
            all_names = list({*(individual_accounts or []), *(sum_accounts or [])})
            if all_names:
                dates = [all_accounts[n].transaction_data["date"] for n in all_names]
                all_dates = pd.concat(dates)
                range_start = start or all_dates.min().strftime("%Y-%m-%d")
                range_end = end or all_dates.max().strftime("%Y-%m-%d")
                layout_kwargs["xaxis_range"] = [range_start, range_end]
        fig.update_layout(**layout_kwargs)
        return _apply_theme(fig, theme)
