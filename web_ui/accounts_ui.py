import shutil
from pathlib import Path

from dash import dcc, html, Input, Output, State, ctx
from omegaconf import OmegaConf

from .assignations_ui import assignations_layout, register_assignations_callbacks
from .i18n import t

_acct_field_style = {"width": "350px", "fontFamily": "monospace"}
_acct_label_style = {"width": "260px", "fontWeight": "bold", "flexShrink": "0"}
_account_status_options = [{"label": s, "value": s} for s in ["OPEN", "CLOSED"]]


def _acct_row(label, component):
    return html.Div(
        [html.Label(label, style=_acct_label_style), component],
        style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
    )


def accounts_layout(all_accounts, account_options: list, default_account: str,
                    lang: str = "en") -> html.Div:
    category_options = [{"label": v, "value": v} for v in sorted({
        acc.category for acc in all_accounts if acc.category
    })]
    family_options = [{"label": v, "value": v} for v in sorted({
        acc.family for acc in all_accounts if acc.family
    })]
    institution_options = [{"label": v, "value": v} for v in sorted({
        acc.institution for acc in all_accounts if acc.institution
    })]

    return html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(
                        id="accounts-tab-selector",
                        options=account_options,
                        value=default_account,
                        clearable=False,
                        style={"width": "300px"},
                    ),
                    html.Button(t(lang, "add_account"), id="add-account-btn", n_clicks=0,
                                style={"marginLeft": "10px"}),
                    html.Button(t(lang, "reload"), id="reload-accounts-btn", n_clicks=0,
                                style={"marginLeft": "10px"}),
                    html.Button(t(lang, "save"), id="save-account-btn", n_clicks=0,
                                style={"marginLeft": "10px"}),
                    html.Button(t(lang, "manage_assignations"), id="manage-assignations-btn",
                                n_clicks=0, style={"marginLeft": "10px"}),
                    html.Span(id="account-save-status",
                              style={"marginLeft": "10px", "color": "green"}),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    dcc.Input(id="new-acct-name",
                              placeholder=t(lang, "account_name_placeholder"),
                              style={"width": "250px"}),
                    html.Button(t(lang, "create"), id="create-account-btn", n_clicks=0,
                                style={"marginLeft": "10px"}),
                    html.Button(t(lang, "cancel"), id="cancel-account-btn", n_clicks=0,
                                style={"marginLeft": "10px"}),
                ],
                id="new-acct-form",
                style={"display": "none", "alignItems": "center", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            _acct_row("Category", dcc.Dropdown(
                                id="acct-category", options=category_options,
                                clearable=True, style=_acct_field_style,
                            )),
                            _acct_row("Family", dcc.Dropdown(
                                id="acct-family", options=family_options,
                                clearable=True, style=_acct_field_style,
                            )),
                            _acct_row("Institution", dcc.Dropdown(
                                id="acct-institution", options=institution_options,
                                clearable=True, style=_acct_field_style,
                            )),
                            _acct_row("Name", dcc.Input(
                                id="acct-name", type="text", style=_acct_field_style,
                            )),
                            _acct_row("Interest rate", dcc.Input(
                                id="acct-interest-rate", type="number",
                                style=_acct_field_style,
                            )),
                            _acct_row("Status", dcc.Dropdown(
                                id="acct-status", options=_account_status_options,
                                clearable=False, style=_acct_field_style,
                            )),
                            _acct_row("Encoding", dcc.Input(
                                id="acct-encoding", type="text", style=_acct_field_style,
                            )),
                            _acct_row("Date format", dcc.Input(
                                id="acct-date-format", type="text", style=_acct_field_style,
                            )),
                            _acct_row("Separator", dcc.Input(
                                id="acct-separator", type="text", style=_acct_field_style,
                            )),
                            _acct_row("Has header row", dcc.Checklist(
                                id="acct-has-header",
                                options=[{"label": "", "value": "yes"}],
                                value=[],
                            )),
                            _acct_row("Column names (one per line)", dcc.Textarea(
                                id="acct-columns-names",
                                style={**_acct_field_style, "height": "120px"},
                            )),
                            _acct_row("Sorting order (one per line)", dcc.Textarea(
                                id="acct-sorting-order",
                                style={**_acct_field_style, "height": "60px"},
                            )),
                            _acct_row("Sorting ascendancy (one per line)", dcc.Textarea(
                                id="acct-sorting-asc",
                                style={**_acct_field_style, "height": "60px"},
                            )),
                            _acct_row("Numerical columns (col,sign per line)", dcc.Textarea(
                                id="acct-numerical-cols",
                                style={**_acct_field_style, "height": "100px"},
                            )),
                            _acct_row("Balance column (col,sign or blank)", dcc.Input(
                                id="acct-balance-col", type="text", style=_acct_field_style,
                            )),
                            _acct_row("Rows selection — filter column", dcc.Input(
                                id="acct-rows-sel-col", type="text", style=_acct_field_style,
                            )),
                            _acct_row("Rows selection — filter value", dcc.Input(
                                id="acct-rows-sel-val", type="text", style=_acct_field_style,
                            )),
                            _acct_row("Initial balance", dcc.Input(
                                id="acct-initial-balance", type="number",
                                style=_acct_field_style,
                            )),
                            _acct_row("Statement day", dcc.Input(
                                id="acct-statement-day", type="text", style=_acct_field_style,
                            )),
                        ],
                        style={"maxWidth": "700px"},
                    ),
                    assignations_layout(all_accounts, lang),
                ],
                style={"display": "flex", "gap": "40px", "alignItems": "flex-start"},
            ),
        ],
        style={"padding": "10px"},
    )


def register_accounts_callbacks(app, all_accounts) -> None:

    @app.callback(
        Output("accounts-tab-selector", "options"),
        Output("accounts-tab-selector", "value"),
        Input("account-selector", "value"),
        State("accounts-tab-selector", "value"),
    )
    def update_accounts_tab_selector(selected_names, current_value):
        selected_names = selected_names or []
        options = [{"label": n, "value": n} for n in selected_names]
        new_value = (
            current_value if current_value in selected_names
            else (selected_names[0] if selected_names else None)
        )
        return options, new_value

    @app.callback(
        Output("acct-name", "value"),
        Output("acct-category", "value"),
        Output("acct-family", "value"),
        Output("acct-institution", "value"),
        Output("acct-interest-rate", "value"),
        Output("acct-status", "value"),
        Output("acct-encoding", "value"),
        Output("acct-date-format", "value"),
        Output("acct-separator", "value"),
        Output("acct-has-header", "value"),
        Output("acct-columns-names", "value"),
        Output("acct-sorting-order", "value"),
        Output("acct-sorting-asc", "value"),
        Output("acct-numerical-cols", "value"),
        Output("acct-balance-col", "value"),
        Output("acct-rows-sel-col", "value"),
        Output("acct-rows-sel-val", "value"),
        Output("acct-initial-balance", "value"),
        Output("acct-statement-day", "value"),
        Input("accounts-tab-selector", "value"),
    )
    def update_account_form(account_name):
        account = all_accounts[account_name]
        conf = OmegaConf.load(Path(account.account_dir) / "config.yaml")

        has_header = ["yes"] if conf.get("has_header_row", True) else []
        columns_names = "\n".join(conf.get("columns_names") or [])
        sorting_order = "\n".join(conf.get("sorting_order") or [])
        sorting_asc = "\n".join(
            "true" if v else "false" for v in (conf.get("sorting_ascendancy") or [])
        )
        numerical_cols = "\n".join(
            f"{c[0]},{c[1]}" for c in (conf.get("numerical_columns") or [])
        )
        bal_raw = conf.get("balance_column")
        balance_col = "" if (bal_raw is None or bal_raw[0] is None) else f"{bal_raw[0]},{bal_raw[1]}"
        rows_sel = conf.get("rows_selection")
        rows_col = rows_sel.get("filter_column", "") if rows_sel else ""
        rows_val = rows_sel.get("filter_value", "") if rows_sel else ""

        return (
            conf.get("name", ""),
            conf.get("category", None),
            conf.get("family", None),
            conf.get("institution", None),
            conf.get("interest_rate", 0),
            conf.get("status", "OPEN"),
            conf.get("encoding", "utf-8"),
            conf.get("date_format", "ISO8601"),
            conf.get("separator", ","),
            has_header,
            columns_names,
            sorting_order,
            sorting_asc,
            numerical_cols,
            balance_col,
            rows_col,
            rows_val,
            conf.get("initial_balance", 0),
            conf.get("statement_day") or "",
        )

    @app.callback(
        Output("account-save-status", "children"),
        Input("save-account-btn", "n_clicks"),
        State("acct-name", "value"),
        State("acct-category", "value"),
        State("acct-family", "value"),
        State("acct-institution", "value"),
        State("acct-interest-rate", "value"),
        State("acct-status", "value"),
        State("acct-encoding", "value"),
        State("acct-date-format", "value"),
        State("acct-separator", "value"),
        State("acct-has-header", "value"),
        State("acct-columns-names", "value"),
        State("acct-sorting-order", "value"),
        State("acct-sorting-asc", "value"),
        State("acct-numerical-cols", "value"),
        State("acct-balance-col", "value"),
        State("acct-rows-sel-col", "value"),
        State("acct-rows-sel-val", "value"),
        State("acct-initial-balance", "value"),
        State("acct-statement-day", "value"),
        State("accounts-tab-selector", "value"),
        State("lang", "data"),
        prevent_initial_call=True,
    )
    def save_account_config(
        _n, name, category, family, institution, interest_rate, status,
        encoding, date_format, separator, has_header, columns_names_str,
        sorting_order_str, sorting_asc_str, numerical_cols_str, balance_col_str,
        rows_col, rows_val, initial_balance, statement_day, account_name, lang,
    ):
        lang = lang or "en"
        account = all_accounts[account_name]
        config_path = Path(account.account_dir) / "config.yaml"
        conf = OmegaConf.load(config_path)

        conf["name"] = name or ""
        conf["category"] = category or None
        conf["family"] = family or None
        conf["institution"] = institution or None
        conf["interest_rate"] = float(interest_rate) if interest_rate is not None else 0
        conf["status"] = status
        conf["encoding"] = encoding or "utf-8"
        conf["date_format"] = date_format or "ISO8601"
        conf["separator"] = separator or ","
        conf["has_header_row"] = "yes" in (has_header or [])

        conf["columns_names"] = [
            c.strip() for c in (columns_names_str or "").splitlines() if c.strip()
        ]
        conf["sorting_order"] = [
            c.strip() for c in (sorting_order_str or "").splitlines() if c.strip()
        ]
        conf["sorting_ascendancy"] = [
            v.strip().lower() == "true"
            for v in (sorting_asc_str or "").splitlines() if v.strip()
        ]

        num_cols = []
        for line in (numerical_cols_str or "").splitlines():
            line = line.strip()
            if "," in line:
                col, sign = line.rsplit(",", 1)
                num_cols.append([col.strip(), int(sign.strip())])
        conf["numerical_columns"] = num_cols

        if balance_col_str and "," in balance_col_str:
            col, sign = balance_col_str.rsplit(",", 1)
            conf["balance_column"] = [col.strip(), int(sign.strip())]
        else:
            conf["balance_column"] = None

        if rows_col and rows_col.strip():
            conf["rows_selection"] = {
                "filter_column": rows_col.strip(),
                "filter_value": (rows_val or "").strip(),
            }
        else:
            conf["rows_selection"] = None

        conf["initial_balance"] = float(initial_balance) if initial_balance is not None else 0
        conf["statement_day"] = (statement_day or "").strip() or None

        OmegaConf.save(conf, config_path)
        return t(lang, "account_saved")

    @app.callback(
        Output("new-acct-form", "style"),
        Input("add-account-btn", "n_clicks"),
        Input("cancel-account-btn", "n_clicks"),
        Input("create-account-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_new_account_form(_add, _cancel, _create):
        visible = {"display": "flex", "alignItems": "center", "marginBottom": "10px"}
        hidden = {"display": "none", "alignItems": "center", "marginBottom": "10px"}
        return visible if ctx.triggered_id == "add-account-btn" else hidden

    @app.callback(
        Output("account-save-status", "children", allow_duplicate=True),
        Input("create-account-btn", "n_clicks"),
        State("new-acct-name", "value"),
        State("lang", "data"),
        prevent_initial_call=True,
    )
    def create_account_callback(_n, account_name, lang):
        lang = lang or "en"
        if not account_name or not account_name.strip():
            return t(lang, "account_name_required")
        name = account_name.strip()
        accounts_dir = Path(all_accounts[0].account_dir).parent
        template_dir = accounts_dir / "_template_account"
        dest_dir = accounts_dir / name.lower()
        if dest_dir.exists():
            return t(lang, "account_exists", name=name.lower())
        shutil.copytree(template_dir, dest_dir)
        shutil.move(str(dest_dir / "_config.yaml"), str(dest_dir / "config.yaml"))
        conf = OmegaConf.load(dest_dir / "config.yaml")
        conf["name"] = name
        OmegaConf.save(conf, dest_dir / "config.yaml")
        return t(lang, "account_created", name=name)

    register_assignations_callbacks(app, all_accounts)
