from pathlib import Path

from dash import Dash, dcc, html, Input, Output

from accounting import AccountFactory
from accounting.account_list import AccountsList
from utils.period_utils import parse_period
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

_filter_bar = html.Div(
    [
        dcc.Dropdown(
            id="category-filter",
            options=_category_options,
            multi=True,
            placeholder="Category…",
            style={"flex": "1"},
        ),
        dcc.Dropdown(
            id="family-filter",
            options=_family_options,
            multi=True,
            placeholder="Family…",
            style={"flex": "2"},
        ),
        dcc.Dropdown(
            id="institution-filter",
            options=_institution_options,
            multi=True,
            placeholder="Institution…",
            style={"flex": "1"},
        ),
        dcc.Dropdown(
            id="account-selector",
            options=_account_options,
            value=_default_accounts,
            multi=True,
            placeholder="Accounts…",
            style={"flex": "3"},
        ),
        dcc.Input(
            id="global-period-input",
            type="text",
            placeholder="Period…",
            debounce=True,
            style={"width": "180px", "fontFamily": "monospace"},
        ),
        html.Button(
            "↺ Reload",
            id="reload-accounts-btn",
            n_clicks=0,
            style={"flexShrink": "0"},
        ),
    ],
    style={
        "display": "flex",
        "gap": "8px",
        "padding": "8px 10px",
        "borderBottom": "1px solid #ddd",
        "backgroundColor": "#f8f8f8",
    },
)

app.layout = html.Div(
    [
        _filter_bar,
        profiles_layout(),
        dcc.Store(id="filtered-accounts", data=_default_accounts),
        dcc.Store(id="global-period", data={"start": None, "end": None}),
        dcc.Tabs(
            [
                dcc.Tab(label="History",
                        children=[history_layout(_account_options, _default_accounts)]),
                dcc.Tab(label="Forecast",
                        children=[forecast_layout()]),
                dcc.Tab(label="Breakdown",
                        children=[breakdown_layout()]),
                dcc.Tab(label="Accounts",
                        children=[accounts_layout(all_accounts, _account_options,
                                                  _default_table_account)]),
                dcc.Tab(label="Transactions",
                        children=[transactions_layout(all_accounts, _account_options,
                                                      _default_table_account)]),
            ]
        ),
    ],
    style={"fontFamily": "sans-serif"},
)


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
    Output("category-filter", "options"),
    Output("category-filter", "value", allow_duplicate=True),
    Output("family-filter", "options"),
    Output("family-filter", "value", allow_duplicate=True),
    Output("institution-filter", "options"),
    Output("institution-filter", "value", allow_duplicate=True),
    Input("reload-accounts-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reload_accounts(_n):
    all_accounts.reload(AccountFactory().accounts)
    category_options = [{"label": v, "value": v} for v in sorted({
        acc.category for acc in all_accounts if acc.category
    })]
    family_options = [{"label": v, "value": v} for v in sorted({
        acc.family for acc in all_accounts if acc.family
    })]
    institution_options = [{"label": v, "value": v} for v in sorted({
        acc.institution for acc in all_accounts if acc.institution
    })]
    return category_options, [], family_options, [], institution_options, []


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
register_forecast_callbacks(app)
register_breakdown_callbacks(app, all_accounts)
register_accounts_callbacks(app, all_accounts)
register_transactions_callbacks(app, all_accounts)
register_profiles_callbacks(app)
