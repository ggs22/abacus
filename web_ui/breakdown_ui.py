import plotly.graph_objects as go
from dash import dcc, html, Input, Output

from accounting.account_list import AccountsList


def breakdown_layout() -> html.Div:
    return html.Div(
        [
            dcc.Graph(id="breakdown-chart", style={"height": "75vh"}),
        ],
        style={"padding": "10px"},
    )


def register_breakdown_callbacks(app, all_accounts) -> None:

    @app.callback(
        Output("breakdown-chart", "figure"),
        Input("global-period", "data"),
        Input("account-selector", "value"),
    )
    def update_breakdown(period_data, selected_names):
        start = (period_data or {}).get("start")
        if not start or not selected_names:
            return go.Figure()
        end = (period_data or {}).get("end")
        filtered = AccountsList([all_accounts[n] for n in selected_names])
        return filtered.barplot(start_date=start, end_date=end)
