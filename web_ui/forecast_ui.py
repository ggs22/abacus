import plotly.graph_objects as go
from dash import dcc, html, Input, Output


def forecast_layout() -> html.Div:
    return html.Div(
        [
            dcc.Checklist(
                id="show-total",
                options=[{"label": "  Show total", "value": "show"}],
                value=["show"],
                style={"marginBottom": "10px"},
            ),
            dcc.Graph(id="forecast-chart", style={"height": "75vh"}),
        ],
        style={"padding": "10px"},
    )


def register_forecast_callbacks(app) -> None:

    @app.callback(
        Output("forecast-chart", "figure"),
        Input("show-total", "value"),
        Input("account-selector", "value"),
        Input("global-period", "data"),
    )
    def update_forecast(show_total_val, selected_names, period_data):
        return go.Figure()
