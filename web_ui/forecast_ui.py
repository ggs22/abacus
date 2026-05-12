import datetime as dt
import json
from pathlib import Path

import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, ALL, MATCH, ctx
from dash.exceptions import PreventUpdate

from accounting.account_list import AccountsList
from .i18n import t
from accounting.forecast_strategies import (
    PlannedTransactionsStrategy, MeanTransactionsStrategy,
    MonteCarloStrategy, ParallelMonteCarloStrategy, NoTransactionsStrategy,
    FixedLoanPaymentForecastStrategy, CreditCardPaymentForecastStrategy,
    ForecastFactory,
)

_CONFIG_PATH = Path(__file__).parent.parent / "accounting" / "accounts" / "forecast_config.json"

_STRATEGIES = {
    "Planned": PlannedTransactionsStrategy,
    "Mean": MeanTransactionsStrategy,
    "MonteCarlo": MonteCarloStrategy,
    "ParallelMonteCarlo": ParallelMonteCarloStrategy,
    "NoTransactions": NoTransactionsStrategy,
}
_STRATEGY_OPTIONS = [{"label": k, "value": k} for k in _STRATEGIES]
_CROSS_TYPE_OPTIONS = [
    {"label": "Fixed Loan Payment", "value": "FixedLoanPayment"},
    {"label": "Credit Card Payment", "value": "CreditCardPayment"},
]

_last_forecasts: dict = {}
_forecast_factory = ForecastFactory()

_DARK_LAYOUT = dict(paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
                    font_color="#cdd6f4")
_LIGHT_LAYOUT = dict(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")


def _apply_theme(fig: go.Figure, theme: str) -> go.Figure:
    if theme == "dark":
        fig.update_layout(template="plotly_dark", **_DARK_LAYOUT)
    else:
        fig.update_layout(template="plotly_white", **_LIGHT_LAYOUT)
    return fig


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {"global": {}, "accounts": {}, "cross_account": []}


def _save_config(config: dict) -> None:
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


def _resolve_sim_date(sim_date_str: str, all_accounts) -> str:
    if sim_date_str and sim_date_str.strip():
        return sim_date_str.strip()
    sim_date = dt.date(1900, 1, 1)
    for acc in all_accounts:
        d = acc.most_recent_date
        if hasattr(d, "date"):
            d = d.date()
        if d > sim_date:
            sim_date = d
    return sim_date.strftime("%Y-%m-%d")


def _cross_row(i: int, entry: dict, account_options: list, lang: str = "en") -> html.Div:
    cross_type = entry.get("type", "CreditCardPayment")
    fixed_style = (
        {"display": "flex", "alignItems": "center"}
        if cross_type == "FixedLoanPayment"
        else {"display": "none"}
    )
    _mono = {"fontFamily": "monospace"}
    return html.Div(
        [
            dcc.Dropdown(
                id={"type": "cross-type", "index": i},
                options=_CROSS_TYPE_OPTIONS,
                value=cross_type,
                clearable=False,
                style={"width": "190px"},
            ),
            html.Span("OP :", style={"margin": "0 4px 0 8px", "flexShrink": "0"}),
            dcc.Dropdown(
                id={"type": "cross-op", "index": i},
                options=account_options,
                value=entry.get("op_account"),
                clearable=False,
                style={"width": "170px"},
            ),
            html.Span("Prêt/CC :" if lang == "fr" else "Loan/CC:", style={"margin": "0 4px 0 8px", "flexShrink": "0"}),
            dcc.Dropdown(
                id={"type": "cross-secondary", "index": i},
                options=account_options,
                value=entry.get("loan_account") or entry.get("cc_account"),
                clearable=False,
                style={"width": "170px"},
            ),
            html.Div(
                [
                    html.Span("Rate:", style={"margin": "0 4px 0 8px", "flexShrink": "0"}),
                    dcc.Input(
                        id={"type": "cross-rate", "index": i}, type="number",
                        value=entry.get("loan_rate", 0.05),
                        style={"width": "75px", **_mono},
                    ),
                    html.Span("Day:", style={"margin": "0 4px 0 8px", "flexShrink": "0"}),
                    dcc.Input(
                        id={"type": "cross-day", "index": i}, type="number",
                        value=entry.get("day_of_month", 1),
                        style={"width": "55px", **_mono},
                    ),
                    html.Span("Amount:", style={"margin": "0 4px 0 8px", "flexShrink": "0"}),
                    dcc.Input(
                        id={"type": "cross-amount", "index": i}, type="number",
                        value=entry.get("payment_amount"),
                        style={"width": "90px", **_mono},
                    ),
                ],
                id={"type": "cross-fixed-params", "index": i},
                style=fixed_style,
            ),
            html.Button(
                "×",
                id={"type": "cross-delete", "index": i},
                n_clicks=0,
                style={"marginLeft": "8px", "color": "red", "flexShrink": "0",
                       "fontWeight": "bold", "border": "none", "background": "none",
                       "cursor": "pointer", "fontSize": "16px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "6px",
               "padding": "6px 8px", "border": "1px solid #e0e0e0",
               "borderRadius": "4px", "flexWrap": "wrap"},
    )


def forecast_layout(all_accounts, lang: str = "en") -> html.Div:
    config = _load_config()
    global_cfg = config.get("global", {})
    accounts_cfg = config.get("accounts", {})

    strategy_rows = [
        html.Div(
            [
                html.Span(acc.name, style={"width": "220px", "fontFamily": "monospace",
                                           "flexShrink": "0"}),
                dcc.Dropdown(
                    id={"type": "forecast-strategy", "account": acc.name},
                    options=_STRATEGY_OPTIONS,
                    value=accounts_cfg.get(acc.name, "Mean"),
                    clearable=False,
                    style={"width": "220px"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
        )
        for acc in all_accounts
    ]

    return html.Div(
        [
            # ── global params bar ──────────────────────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(t(lang, "end_date"), style={"fontSize": "12px"}),
                            dcc.Input(
                                id="forecast-end-date", type="text", debounce=True,
                                value=global_cfg.get("end_date", ""),
                                placeholder="YYYY-MM-DD",
                                style={"width": "120px", "fontFamily": "monospace"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "2px"},
                    ),
                    html.Div(
                        [
                            html.Label(t(lang, "simulation_date"), style={"fontSize": "12px"}),
                            dcc.Input(
                                id="forecast-sim-date", type="text", debounce=True,
                                value=global_cfg.get("simulation_date", ""),
                                placeholder="YYYY-MM-DD",
                                style={"width": "140px", "fontFamily": "monospace"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "2px"},
                    ),
                    html.Div(
                        [
                            html.Label(t(lang, "mc_iterations"), style={"fontSize": "12px"}),
                            dcc.Input(
                                id="forecast-mc-iterations", type="number", min=1,
                                value=global_cfg.get("mc_iterations", 100),
                                style={"width": "90px"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "2px"},
                    ),
                    html.Div(
                        [
                            html.Button(t(lang, "run_forecast"), id="run-forecast-btn",
                                        n_clicks=0, style={"fontWeight": "bold"}),
                            html.Span("", id="forecast-status",
                                      style={"marginLeft": "10px", "color": "green",
                                             "fontFamily": "monospace"}),
                        ],
                        style={"display": "flex", "alignItems": "center",
                               "alignSelf": "flex-end"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "alignItems": "flex-end",
                       "marginBottom": "10px", "flexWrap": "wrap"},
            ),
            # ── per-account strategy table ─────────────────────────────────
            html.Details(
                [
                    html.Summary(
                        t(lang, "per_account_strategies"),
                        style={"fontWeight": "bold", "cursor": "pointer",
                               "marginBottom": "6px", "userSelect": "none"},
                    ),
                    html.Div(strategy_rows,
                             style={"maxHeight": "240px", "overflowY": "auto",
                                    "padding": "4px 0"}),
                ],
                open=True,
                style={"marginBottom": "10px", "border": "1px solid #ddd",
                       "borderRadius": "4px", "padding": "8px"},
            ),
            # ── account selector + sum group ───────────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(t(lang, "forecast_accounts"),
                                       style={"fontWeight": "bold", "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="forecast-account-selector",
                                options=[{"label": acc.name, "value": acc.name}
                                         for acc in all_accounts],
                                value=[acc.name for acc in all_accounts],
                                multi=True,
                                placeholder=t(lang, "ph_select_forecast_accounts"),
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label(t(lang, "sum_group"),
                                       style={"fontWeight": "bold", "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="forecast-sum-selector",
                                options=[],
                                value=[],
                                multi=True,
                                placeholder=t(lang, "ph_select_forecast_sum"),
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "marginBottom": "8px"},
            ),
            # ── chart ──────────────────────────────────────────────────────
            dcc.Store(id="forecast-run-trigger", data=None),
            dcc.Graph(id="forecast-chart", style={"height": "55vh"}),
            # ── cross-account section ──────────────────────────────────────
            html.Div(
                id="cross-account-section",
                style={"display": "none"},
                children=[
                    html.Hr(),
                    html.Div(
                        [
                            html.Span(t(lang, "cross_account_strategies"),
                                      style={"fontWeight": "bold", "fontSize": "15px"}),
                            html.Button(t(lang, "add"), id="add-cross-btn", n_clicks=0,
                                        style={"marginLeft": "12px"}),
                            html.Button(t(lang, "run"), id="run-cross-btn", n_clicks=0,
                                        style={"marginLeft": "6px", "fontWeight": "bold"}),
                            html.Span("", id="cross-status",
                                      style={"marginLeft": "10px", "color": "green",
                                             "fontFamily": "monospace"}),
                        ],
                        style={"display": "flex", "alignItems": "center",
                               "marginBottom": "8px"},
                    ),
                    dcc.Store(id="cross-account-store",
                              data=config.get("cross_account", [])),
                    html.Div(id="cross-account-rows"),
                ],
            ),
        ],
        style={"padding": "10px"},
    )


def register_forecast_callbacks(app, all_accounts) -> None:
    account_options = [{"label": acc.name, "value": acc.name} for acc in all_accounts]

    @app.callback(
        Output("forecast-account-selector", "options"),
        Output("forecast-account-selector", "value"),
        Input("filtered-accounts", "data"),
    )
    def update_forecast_selector(filtered_names):
        options = [{"label": n, "value": n} for n in filtered_names]
        return options, filtered_names

    @app.callback(
        Output("forecast-run-trigger", "data"),
        Output("forecast-status", "children"),
        Output("cross-account-section", "style"),
        Output("forecast-sum-selector", "options"),
        Input("run-forecast-btn", "n_clicks"),
        State({"type": "forecast-strategy", "account": ALL}, "value"),
        State({"type": "forecast-strategy", "account": ALL}, "id"),
        State("forecast-end-date", "value"),
        State("forecast-sim-date", "value"),
        State("forecast-mc-iterations", "value"),
        State("forecast-account-selector", "value"),
        State("lang", "data"),
        prevent_initial_call=True,
    )
    def run_forecast(_, strategies, ids, end_date, sim_date_str, mc_iters, selected_names, lang):
        global _last_forecasts
        lang = lang or "en"

        selected_names = selected_names or []
        sim_date = _resolve_sim_date(sim_date_str, all_accounts)

        if end_date and end_date.strip():
            try:
                predicted_days = (
                    dt.date.fromisoformat(end_date.strip())
                    - dt.date.fromisoformat(sim_date)
                ).days
            except ValueError:
                return None, t(lang, "forecast_err_date"), {"display": "none"}, []
        else:
            predicted_days = 365

        if predicted_days <= 0:
            return None, t(lang, "forecast_err_past"), {"display": "none"}, []

        strategy_map = {d["account"]: s for d, s in zip(ids, strategies)}
        mc_iterations = int(mc_iters or 100)

        pred_args = {
            "predicted_days": predicted_days,
            "simulation_date": sim_date,
            "mc_iterations": mc_iterations,
        }

        requested = [
            (all_accounts[name], _STRATEGIES[strategy_map.get(name, "Mean")](), pred_args)
            for name in selected_names
            if name in strategy_map
            and name in all_accounts.accounts
            and all_accounts[name].status == "OPEN"
        ]

        _last_forecasts = _forecast_factory(requested)

        config = _load_config()
        config["global"] = {
            "end_date": end_date or "",
            "simulation_date": sim_date_str or "",
            "mc_iterations": mc_iterations,
        }
        config["accounts"] = strategy_map
        _save_config(config)

        sum_options = [{"label": n, "value": n} for n in _last_forecasts]
        trigger = {"ts": dt.datetime.now().isoformat()}
        status = t(lang, "forecast_done", n=len(_last_forecasts))
        return trigger, status, {}, sum_options

    @app.callback(
        Output("forecast-chart", "figure"),
        Input("forecast-run-trigger", "data"),
        Input("forecast-sum-selector", "value"),
        State("lang", "data"),
        State("theme", "data"),
    )
    def render_forecast_chart(trigger, sum_accounts, lang, theme):
        lang = lang or "en"
        if not _last_forecasts:
            return _apply_theme(go.Figure(), theme or "light")
        accounts = AccountsList([
            all_accounts[n] for n in _last_forecasts
            if n in all_accounts.accounts
        ])
        fig = accounts.plot_forecasts(
            forecasts=_last_forecasts,
            sum_accounts=sum_accounts or [],
            title=t(lang, "chart_forecast_title"),
        )
        return _apply_theme(fig, theme or "light")

    @app.callback(
        Output("cross-account-rows", "children"),
        Input("cross-account-store", "data"),
        State("lang", "data"),
    )
    def render_cross_rows(store_data, lang):
        lang = lang or "en"
        return [_cross_row(i, e, account_options, lang) for i, e in enumerate(store_data or [])]

    @app.callback(
        Output({"type": "cross-fixed-params", "index": MATCH}, "style"),
        Input({"type": "cross-type", "index": MATCH}, "value"),
    )
    def toggle_fixed_params(cross_type):
        if cross_type == "FixedLoanPayment":
            return {"display": "flex", "alignItems": "center"}
        return {"display": "none"}

    @app.callback(
        Output("cross-account-store", "data"),
        Input("add-cross-btn", "n_clicks"),
        State("cross-account-store", "data"),
        prevent_initial_call=True,
    )
    def add_cross_entry(_, store_data):
        entries = list(store_data or [])
        entries.append({"type": "CreditCardPayment", "op_account": None, "cc_account": None})
        return entries

    @app.callback(
        Output("cross-account-store", "data", allow_duplicate=True),
        Input({"type": "cross-delete", "index": ALL}, "n_clicks"),
        State("cross-account-store", "data"),
        prevent_initial_call=True,
    )
    def delete_cross_entry(n_clicks_list, store_data):
        if not any(n_clicks_list):
            raise PreventUpdate
        triggered = ctx.triggered_id
        if triggered is None:
            raise PreventUpdate
        entries = list(store_data or [])
        idx = triggered["index"]
        if 0 <= idx < len(entries):
            entries.pop(idx)
        return entries

    @app.callback(
        Output("forecast-chart", "figure", allow_duplicate=True),
        Output("cross-status", "children"),
        Output("cross-account-store", "data", allow_duplicate=True),
        Input("run-cross-btn", "n_clicks"),
        State({"type": "cross-type", "index": ALL}, "value"),
        State({"type": "cross-op", "index": ALL}, "value"),
        State({"type": "cross-secondary", "index": ALL}, "value"),
        State({"type": "cross-rate", "index": ALL}, "value"),
        State({"type": "cross-day", "index": ALL}, "value"),
        State({"type": "cross-amount", "index": ALL}, "value"),
        State("cross-account-store", "data"),
        State("forecast-sum-selector", "value"),
        State("lang", "data"),
        State("theme", "data"),
        prevent_initial_call=True,
    )
    def run_cross_forecasts(_, types, ops, secondaries, rates, days, amounts, store_data,
                            sum_accounts, lang, theme):
        lang = lang or "en"
        plotly_template = "plotly_dark" if theme == "dark" else "plotly_white"
        global _last_forecasts

        if not _last_forecasts:
            raise PreventUpdate

        requested = []
        new_store = []
        errors = []

        for cross_type, op, secondary, rate, day, amount in zip(
            types, ops, secondaries, rates, days, amounts
        ):
            if not op or not secondary:
                continue
            missing = [n for n in (op, secondary) if n not in _last_forecasts]
            if missing:
                errors.append(f"{', '.join(missing)} not in forecasts")
                continue

            if cross_type == "FixedLoanPayment":
                new_store.append({
                    "type": cross_type, "op_account": op,
                    "loan_account": secondary,
                    "loan_rate": rate or 0,
                    "day_of_month": int(day or 1),
                    "payment_amount": float(amount) if amount is not None else None,
                })
                kwargs = {
                    "op_forecast": _last_forecasts[op],
                    "loan_account": all_accounts[secondary],
                    "loan_forecast": _last_forecasts[secondary],
                    "loan_rate": float(rate or 0),
                    "day_of_month": int(day or 1),
                    "payment_amount": float(amount) if amount is not None else None,
                }
                requested.append(
                    (all_accounts[op], FixedLoanPaymentForecastStrategy(), kwargs)
                )
            elif cross_type == "CreditCardPayment":
                new_store.append({
                    "type": cross_type, "op_account": op, "cc_account": secondary,
                })
                kwargs = {
                    "op_forecast": _last_forecasts[op],
                    "cc_account": all_accounts[secondary],
                    "cc_forecast": _last_forecasts[secondary],
                }
                requested.append(
                    (all_accounts[op], CreditCardPaymentForecastStrategy(), kwargs)
                )

        if not requested:
            msg = "Nothing to run." + (f" Errors: {'; '.join(errors)}" if errors else "")
            raise PreventUpdate

        _last_forecasts = _forecast_factory(requested, _last_forecasts)

        config = _load_config()
        config["cross_account"] = new_store
        _save_config(config)

        accounts = AccountsList([
            all_accounts[n] for n in _last_forecasts if n in all_accounts.accounts
        ])
        fig = accounts.plot_forecasts(
            forecasts=_last_forecasts,
            sum_accounts=sum_accounts or [],
            title=t(lang, "chart_forecast_title"),
        )
        _apply_theme(fig, theme or "light")

        status = t(lang, "cross_done", n=len(requested))
        if errors:
            status += f" Skipped: {'; '.join(errors)}."
        return fig, status, new_store
