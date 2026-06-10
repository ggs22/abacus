from pathlib import Path

from dash import Dash, dcc, html, Input, Output, State

from backend import AccountFactory
from backend.account_list import AccountsList
from utils.period_utils import parse_period
from .gui_config import load_gui_config, save_gui_config
from .i18n import t
from .history_ui import history_layout, register_history_callbacks
from .forecast_ui import forecast_layout, register_forecast_callbacks
from .breakdown_ui import breakdown_layout, register_breakdown_callbacks
from .accounts_ui import accounts_layout, register_accounts_callbacks
from .transactions_ui import transactions_layout, register_transactions_callbacks
from .profiles_ui import profiles_layout, register_profiles_callbacks

# ── load accounts ─────────────────────────────────────────────────────────────
account_factory = AccountFactory()
all_accounts: AccountsList = account_factory.accounts

# ── dash app ──────────────────────────────────────────────────────────────────
app = Dash(__name__, assets_folder=str(Path(__file__).parent / "assets"))

# Immediately apply theme to <html> on load/change to avoid flash
app.clientside_callback(
    """function(theme) {
        var t = theme || 'light';
        document.documentElement.dataset.theme = t;
        localStorage.setItem('abacus-theme', t);
        return null;
    }""",
    Output("_theme-sync", "data"),
    Input("theme", "data"),
)


def serve_layout():
    config = load_gui_config()
    lang  = config.get("language", "en")
    theme = config.get("theme", "light")

    _account_options = [{"label": acc.name, "value": acc.name} for acc in all_accounts]
    _default_accounts = [acc.name for acc in all_accounts]
    _default_table_account = all_accounts[0].name

    _category_options = [{"label": v, "value": v} for v in sorted({
        acc.category for acc in all_accounts if acc.category
    })]
    _family_options = [{"label": v, "value": v} for v in sorted({
        acc.family for acc in all_accounts if acc.family
    })]
    _institution_options = [{"label": v, "value": v} for v in sorted({
        acc.institution for acc in all_accounts if acc.institution
    })]

    _settings_btn = html.Div(
        [
            html.Button("⚙", id="settings-btn", n_clicks=0, title="Settings"),
            html.Div(
                id="settings-panel",
                style={
                    "display": "none",
                    "position": "absolute",
                    "right": "0",
                    "top": "40px",
                    "zIndex": "1000",
                    "border": "1px solid",
                    "borderRadius": "6px",
                    "padding": "14px 16px",
                    "minWidth": "170px",
                },
                children=[
                    html.Div(t(lang, "language"),
                             style={"fontWeight": "bold", "fontSize": "12px",
                                    "marginBottom": "6px"}),
                    dcc.RadioItems(
                        id="lang-toggle",
                        options=[{"label": " EN", "value": "en"},
                                 {"label": " FR", "value": "fr"}],
                        value=lang,
                        labelStyle={"display": "block", "padding": "3px 0",
                                    "cursor": "pointer"},
                        inputStyle={"marginRight": "6px"},
                    ),
                    html.Hr(style={"margin": "10px 0"}),
                    html.Div(t(lang, "theme"),
                             style={"fontWeight": "bold", "fontSize": "12px",
                                    "marginBottom": "6px"}),
                    dcc.RadioItems(
                        id="theme-toggle",
                        options=[
                            {"label": f" {t(lang, 'theme_light')}", "value": "light"},
                            {"label": f" {t(lang, 'theme_dark')}",  "value": "dark"},
                        ],
                        value=theme,
                        labelStyle={"display": "block", "padding": "3px 0",
                                    "cursor": "pointer"},
                        inputStyle={"marginRight": "6px"},
                    ),
                ],
            ),
        ],
        style={"position": "relative", "flexShrink": "0"},
    )

    _filter_bar = html.Div(
        [
            dcc.Dropdown(
                id="category-filter",
                options=_category_options,
                multi=True,
                placeholder=t(lang, "ph_category"),
                style={"flex": "1"},
            ),
            dcc.Dropdown(
                id="family-filter",
                options=_family_options,
                multi=True,
                placeholder=t(lang, "ph_family"),
                style={"flex": "2"},
            ),
            dcc.Dropdown(
                id="institution-filter",
                options=_institution_options,
                multi=True,
                placeholder=t(lang, "ph_institution"),
                style={"flex": "1"},
            ),
            dcc.Dropdown(
                id="account-selector",
                options=_account_options,
                value=_default_accounts,
                multi=True,
                placeholder=t(lang, "ph_accounts"),
                style={"flex": "3"},
            ),
            dcc.Input(
                id="global-period-input",
                type="text",
                value="::",
                placeholder=t(lang, "ph_period"),
                debounce=True,
                style={"width": "180px", "fontFamily": "monospace"},
            ),
            _settings_btn,
        ],
        style={
            "display": "flex",
            "gap": "8px",
            "padding": "8px 10px",
            "alignItems": "center",
            "borderBottom": "1px solid var(--border, #ddd)",
            "backgroundColor": "var(--bg2, #f8f8f8)",
        },
    )

    return html.Div(
        [
            dcc.Location(id="lang-location", refresh=True),
            dcc.Store(id="lang",         data=lang),
            dcc.Store(id="theme",        data=theme),
            dcc.Store(id="_theme-sync"),
            _filter_bar,
            profiles_layout(lang),
            dcc.Store(id="filtered-accounts", data=_default_accounts),
            dcc.Store(id="global-period", data={"start": None, "end": None}),
            dcc.Tabs(
                [
                    dcc.Tab(label=t(lang, "tab_history"),
                            children=[history_layout(_account_options, _default_accounts, lang)]),
                    dcc.Tab(label=t(lang, "tab_forecast"),
                            children=[forecast_layout(all_accounts, lang)]),
                    dcc.Tab(label=t(lang, "tab_breakdown"),
                            children=[breakdown_layout(lang)]),
                    dcc.Tab(label=t(lang, "tab_accounts"),
                            children=[accounts_layout(all_accounts, _account_options,
                                                      _default_table_account, lang)]),
                    dcc.Tab(label=t(lang, "tab_transactions"),
                            children=[transactions_layout(all_accounts, _account_options,
                                                          _default_table_account, lang)]),
                ]
            ),
        ],
        style={"fontFamily": "sans-serif", "backgroundColor": "var(--bg)",
               "color": "var(--text)", "minHeight": "100vh"},
    )


app.layout = serve_layout


# ── settings panel toggle ─────────────────────────────────────────────────────

_PANEL_HIDDEN = {"display": "none", "position": "absolute", "right": "0", "top": "40px",
                 "zIndex": "1000", "border": "1px solid", "borderRadius": "6px",
                 "padding": "14px 16px", "minWidth": "170px"}
_PANEL_SHOWN  = {**_PANEL_HIDDEN, "display": "block"}


@app.callback(
    Output("settings-panel", "style"),
    Input("settings-btn", "n_clicks"),
    State("settings-panel", "style"),
    prevent_initial_call=True,
)
def toggle_settings(_, style):
    return _PANEL_HIDDEN if (style or {}).get("display") != "none" else _PANEL_SHOWN


# ── save lang + theme → reload page ──────────────────────────────────────────

@app.callback(
    Output("lang-location", "href"),
    Input("lang-toggle",  "value"),
    Input("theme-toggle", "value"),
    prevent_initial_call=True,
)
def save_settings(lang, theme):
    config = load_gui_config()
    config["language"] = lang
    config["theme"]    = theme
    save_gui_config(config)
    return "/"


# ── hierarchy filter → store ──────────────────────────────────────────────────

@app.callback(
    Output("filtered-accounts", "data"),
    Input("category-filter", "value"),
    Input("family-filter", "value"),
    Input("institution-filter", "value"),
)
def update_filtered_accounts(categories, families, institutions):
    return [
        acc.name for acc in all_accounts
        if (not categories or acc.category in categories)
        and (not families or acc.family in families)
        and (not institutions or acc.institution in institutions)
    ]



@app.callback(
    Output("global-period", "data"),
    Input("global-period-input", "value"),
)
def update_global_period(period):
    if not period:
        return {"start": None, "end": None}
    try:
        start, end = parse_period(period)
        return {"start": start, "end": end}
    except ValueError:
        return {"start": None, "end": None}


register_history_callbacks(app, all_accounts)
register_forecast_callbacks(app, all_accounts)
register_breakdown_callbacks(app, all_accounts)
register_accounts_callbacks(app, all_accounts)
register_transactions_callbacks(app, all_accounts)
register_profiles_callbacks(app)
