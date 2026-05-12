import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

from accounting.account_list import AccountsList
from accounting.plotting import make_balance_trace


def history_layout(account_options: list, default_accounts: list) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Individual accounts",
                                       style={"fontWeight": "bold", "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="history-individual-selector",
                                options=account_options,
                                value=default_accounts,
                                multi=True,
                                placeholder="Select accounts…",
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label("Sum group",
                                       style={"fontWeight": "bold", "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="history-sum-selector",
                                options=account_options,
                                value=[],
                                multi=True,
                                placeholder="Select accounts to sum…",
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
    )
    def update_history(individual_accounts, sum_accounts, period_data):
        fig = go.Figure()
        if individual_accounts:
            individual = AccountsList([all_accounts[n] for n in individual_accounts])
            for account in individual:
                fig.add_trace(make_balance_trace(account))
        if sum_accounts:
            sum_list = AccountsList([all_accounts[n] for n in sum_accounts])
            total = sum_list.balance_column
            label = "Sum:<br>" + "<br>".join(sum_accounts)
            fig.add_trace(go.Scatter(
                x=total.index, y=total["balance_total"],
                name=label, mode="lines",
                line=dict(color=sum_list.color, dash="dash"),
            ))
        layout_kwargs = dict(title="Balance history", xaxis_title="Date",
                             yaxis_title="Balance", showlegend=True)
        start = (period_data or {}).get("start")
        end = (period_data or {}).get("end")
        if start and end:
            layout_kwargs["xaxis_range"] = [start, end]
        fig.update_layout(**layout_kwargs)
        return fig
