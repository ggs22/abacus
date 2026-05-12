import json
from pathlib import Path

from dash import html, dcc, Input, Output, State, ALL, ctx

from .i18n import t


def _common_path() -> Path:
    return Path(__file__).parent.parent / "accounting" / "accounts" / "common_assignation.json"


def _account_path(account_dir: str) -> Path:
    return Path(account_dir) / "assignations.json"


def _load(scope: str, all_accounts) -> dict:
    if scope == "__global__":
        path = _common_path()
    else:
        path = _account_path(all_accounts[scope].account_dir)
    with open(path) as f:
        return json.load(f)


def _save(scope: str, data: dict, all_accounts) -> None:
    if scope == "__global__":
        path = _common_path()
    else:
        path = _account_path(all_accounts[scope].account_dir)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def assignations_layout(all_accounts, lang: str = "en") -> html.Div:
    scope_options = [{"label": "Global", "value": "__global__"}] + [
        {"label": acc.name, "value": acc.name} for acc in all_accounts
    ]

    return html.Div(
        id="assignations-panel",
        style={"display": "none"},
        children=[
            dcc.Store(id="assign-data"),
            html.Div([
                html.Label(t(lang, "scope"), style={"fontWeight": "bold", "marginRight": "8px",
                                                    "flexShrink": "0"}),
                dcc.Dropdown(
                    id="assign-scope",
                    options=scope_options,
                    value="__global__",
                    clearable=False,
                    style={"width": "250px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "14px"}),
            html.Div([
                # ── left: codes list ─────────────────────────────────────────
                html.Div([
                    html.Div(t(lang, "codes"),
                             style={"fontWeight": "bold", "marginBottom": "6px"}),
                    dcc.RadioItems(
                        id="assign-code-selector",
                        options=[],
                        value=None,
                        style={"maxHeight": "380px", "overflowY": "auto"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "block", "padding": "3px 0",
                                    "fontFamily": "monospace"},
                    ),
                    html.Div([
                        dcc.Input(
                            id="assign-new-code-input",
                            placeholder=t(lang, "ph_new_code"),
                            debounce=False,
                            style={"width": "150px", "marginRight": "6px",
                                   "fontFamily": "monospace"},
                        ),
                        html.Button(t(lang, "add"), id="assign-add-code-btn", n_clicks=0),
                        html.Button(
                            t(lang, "delete"),
                            id="assign-del-code-btn",
                            n_clicks=0,
                            style={"marginLeft": "6px", "color": "red"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"}),
                ], style={"width": "220px", "flexShrink": "0", "marginRight": "30px"}),

                # ── right: keywords panel ─────────────────────────────────────
                html.Div([
                    html.Div(
                        id="assign-code-label",
                        style={"fontWeight": "bold", "marginBottom": "6px"},
                    ),
                    html.Div(
                        id="assign-keywords-list",
                        style={"maxHeight": "380px", "overflowY": "auto",
                               "minWidth": "260px"},
                    ),
                    html.Div([
                        dcc.Input(
                            id="assign-new-kw-input",
                            placeholder=t(lang, "ph_new_keyword"),
                            debounce=False,
                            style={"width": "220px", "marginRight": "6px",
                                   "fontFamily": "monospace"},
                        ),
                        html.Button(t(lang, "add"), id="assign-add-kw-btn", n_clicks=0),
                    ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"}),
                    html.Div([
                        html.Button(
                            t(lang, "save"),
                            id="assign-save-btn",
                            n_clicks=0,
                            style={"marginTop": "12px"},
                        ),
                        html.Span(
                            id="assign-save-status",
                            style={"marginLeft": "10px", "color": "green",
                                   "fontFamily": "monospace"},
                        ),
                    ]),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "alignItems": "flex-start"}),
        ],
    )


def register_assignations_callbacks(app, all_accounts) -> None:

    @app.callback(
        Output("assignations-panel", "style"),
        Input("manage-assignations-btn", "n_clicks"),
        State("assignations-panel", "style"),
        prevent_initial_call=True,
    )
    def toggle_panel(_, style):
        hidden = {"display": "none"}
        visible = {"display": "block"}
        return hidden if style.get("display") != "none" else visible

    @app.callback(
        Output("assign-data", "data"),
        Input("assign-scope", "value"),
    )
    def load_scope(scope):
        return _load(scope, all_accounts)

    @app.callback(
        Output("assign-code-selector", "options"),
        Output("assign-code-selector", "value"),
        Input("assign-data", "data"),
        State("assign-code-selector", "value"),
    )
    def update_codes(data, current):
        if not data:
            return [], None
        codes = list(data.keys())
        options = [{"label": c, "value": c} for c in codes]
        value = current if current in codes else (codes[0] if codes else None)
        return options, value

    @app.callback(
        Output("assign-data", "data", allow_duplicate=True),
        Output("assign-new-code-input", "value"),
        Input("assign-add-code-btn", "n_clicks"),
        State("assign-new-code-input", "value"),
        State("assign-data", "data"),
        State("assign-scope", "value"),
        prevent_initial_call=True,
    )
    def add_code(_, new_code, data, scope):
        name = (new_code or "").strip()
        if not name or name in data:
            return data, ""
        data[name] = []
        _save(scope, data, all_accounts)
        return data, ""

    @app.callback(
        Output("assign-data", "data", allow_duplicate=True),
        Output("assign-code-selector", "value", allow_duplicate=True),
        Input("assign-del-code-btn", "n_clicks"),
        State("assign-code-selector", "value"),
        State("assign-data", "data"),
        State("assign-scope", "value"),
        prevent_initial_call=True,
    )
    def delete_code(_, selected, data, scope):
        if selected and selected in data:
            del data[selected]
        _save(scope, data, all_accounts)
        codes = list(data.keys())
        return data, codes[0] if codes else None

    @app.callback(
        Output("assign-code-label", "children"),
        Output("assign-keywords-list", "children"),
        Input("assign-code-selector", "value"),
        Input("assign-data", "data"),
    )
    def update_keywords_panel(selected, data):
        if not selected or not data:
            return "Select a code", []

        keywords = data.get(selected, [])
        label = f"Keywords for: {selected}"

        rows = [
            html.Div([
                html.Span(kw, style={"fontFamily": "monospace", "marginRight": "8px",
                                     "flex": "1"}),
                html.Button(
                    "×",
                    id={"type": "assign-del-kw", "index": i},
                    n_clicks=0,
                    style={"border": "none", "background": "none", "color": "red",
                           "cursor": "pointer", "fontSize": "16px", "lineHeight": "1",
                           "padding": "0 4px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "padding": "2px 0"})
            for i, kw in enumerate(keywords)
        ]

        return label, rows

    @app.callback(
        Output("assign-data", "data", allow_duplicate=True),
        Output("assign-new-kw-input", "value"),
        Input("assign-add-kw-btn", "n_clicks"),
        State("assign-new-kw-input", "value"),
        State("assign-code-selector", "value"),
        State("assign-data", "data"),
        State("assign-scope", "value"),
        prevent_initial_call=True,
    )
    def add_keyword(_, new_kw, selected, data, scope):
        kw = (new_kw or "").strip()
        if not kw or not selected:
            return data, ""
        if kw not in data[selected]:
            data[selected].append(kw)
            _save(scope, data, all_accounts)
        return data, ""

    @app.callback(
        Output("assign-data", "data", allow_duplicate=True),
        Input({"type": "assign-del-kw", "index": ALL}, "n_clicks"),
        State("assign-code-selector", "value"),
        State("assign-data", "data"),
        State("assign-scope", "value"),
        prevent_initial_call=True,
    )
    def delete_keyword(n_clicks_list, selected, data, scope):
        if not any(n_clicks_list) or not selected:
            return data
        triggered = ctx.triggered_id
        if triggered is None:
            return data
        idx = triggered["index"]
        keywords = data[selected]
        if 0 <= idx < len(keywords):
            data[selected] = keywords[:idx] + keywords[idx + 1:]
            _save(scope, data, all_accounts)
        return data

    @app.callback(
        Output("assign-save-status", "children"),
        Input("assign-save-btn", "n_clicks"),
        State("assign-scope", "value"),
        State("assign-data", "data"),
        State("lang", "data"),
        prevent_initial_call=True,
    )
    def save(_, scope, data, lang):
        _save(scope, data, all_accounts)
        return t(lang or "en", "account_saved")
