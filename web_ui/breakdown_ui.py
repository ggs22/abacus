import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

from accounting.account_list import AccountsList
from .i18n import t

_DARK_LAYOUT = dict(paper_bgcolor="#1e1e2e", plot_bgcolor="#181825", font_color="#cdd6f4")
_LIGHT_LAYOUT = dict(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")


def _apply_theme(fig: go.Figure, theme: str) -> go.Figure:
    if theme == "dark":
        fig.update_layout(template="plotly_dark", **_DARK_LAYOUT)
    else:
        fig.update_layout(template="plotly_white", **_LIGHT_LAYOUT)
    return fig


def breakdown_layout(lang: str = "en") -> html.Div:
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
        State("lang", "data"),
        State("theme", "data"),
    )
    def update_breakdown(period_data, selected_names, lang, theme):
        lang = lang or "en"
        theme = theme or "light"
        start = (period_data or {}).get("start")
        if not start or not selected_names:
            return _apply_theme(go.Figure(), theme)
        end = (period_data or {}).get("end")
        filtered = AccountsList([all_accounts[n] for n in selected_names])
        fig = filtered.barplot(start_date=start, end_date=end,
                               title=t(lang, "chart_breakdown_title"))
        return _apply_theme(fig, theme)
