import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd
from dash import html, dcc, dash_table, Input, Output, State, ctx

from accounting.account_list import AccountsList
from .i18n import t


_FLOAT_TOLERANCE = 1e-2
_FLOAT_RE = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_OP_RE = r"<=|>=|<|>"
_OP_FN: dict[str, Callable] = {
    "<":  lambda s, v: s.lt(v),
    "<=": lambda s, v: s.le(v),
    ">":  lambda s, v: s.gt(v),
    ">=": lambda s, v: s.ge(v),
}
_REVERSE_OP = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
_FloatMask = Callable[[pd.Series], pd.Series]

_TABLE_CELL = {"textAlign": "left", "padding": "4px 8px",
               "fontFamily": "monospace", "fontSize": "13px"}
_TOTALS_STYLE = {"fontFamily": "monospace", "fontSize": "13px",
                 "padding": "6px 8px", "borderTop": "2px solid #ccc",
                 "marginTop": "4px"}
_NA_HIGHLIGHT = [{"if": {"filter_query": '{code} = "na"'},
                  "backgroundColor": "#fff3cd"}]
_SHOWN_ROW = {"display": "flex", "alignItems": "center", "marginBottom": "10px"}
_HIDDEN = {"display": "none"}


def _all_codes(all_accounts) -> list[str]:
    codes = set()
    for acc in all_accounts:
        codes.update(acc.transaction_data["code"].dropna().unique())
    return sorted(codes)


def _text_cols(df: pd.DataFrame) -> list[str]:
    excluded = {"code", "date", "_idx"}
    return [c for c in df.select_dtypes(include="object").columns if c not in excluded]


def _parse_float_filter(s: str) -> tuple[_FloatMask | None, bool]:
    s = (s or "").strip()
    use_abs = False
    if s.lower().startswith("abs(") and s.endswith(")"):
        s = s[4:-1].strip()
        use_abs = True
    if not s:
        return None, False

    # Range: lo <op> x <op> hi  e.g. "0<x<100" or "0<=x<=100"
    m = re.fullmatch(
        rf"({_FLOAT_RE})\s*({_OP_RE})\s*x\s*({_OP_RE})\s*({_FLOAT_RE})",
        s, re.IGNORECASE,
    )
    if m:
        lo, lo_op, hi_op, hi = m.groups()
        lo_val, hi_val = float(lo), float(hi)
        lo_fn = _OP_FN[_REVERSE_OP[lo_op]]
        hi_fn = _OP_FN[hi_op]
        return lambda series, lo_fn=lo_fn, hi_fn=hi_fn, lo_val=lo_val, hi_val=hi_val: \
            lo_fn(series, lo_val) & hi_fn(series, hi_val), use_abs

    # Single operator: >100, >=100, <100, <=100
    m = re.fullmatch(rf"({_OP_RE})\s*({_FLOAT_RE})", s)
    if m:
        op, val = m.groups()
        val = float(val)
        fn = _OP_FN[op]
        return lambda series, fn=fn, val=val: fn(series, val), use_abs

    # Exact match
    try:
        val = float(s)
        return lambda series, val=val: series.sub(val).abs().lt(_FLOAT_TOLERANCE), use_abs
    except ValueError:
        return None, False


def _codes_for(account) -> list[str]:
    common_path = Path(account.account_dir).parent / "common_assignation.json"
    with open(common_path) as f:
        common = list(json.load(f).keys())
    acc_path = Path(account.account_dir) / "assignations.json"
    with open(acc_path) as f:
        acc_codes = list(json.load(f).keys())
    return sorted(set(acc_codes + common + ["na"]))


def _fmt(account, df: pd.DataFrame) -> tuple:
    """Format a DataFrame for DataTable output, preserving original indices in _idx."""
    df = df.copy()
    df["_idx"] = df.index
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("date", ascending=False)

    visible = [c for c in df.columns if c != "_idx"]
    columns = (
        [{"name": c, "id": c, "editable": c == "code",
          "presentation": "dropdown" if c == "code" else "input"}
         for c in visible]
        + [{"name": "_idx", "id": "_idx", "hideable": True}]
    )
    dropdown = {"code": {"options": [{"label": c, "value": c}
                                      for c in _codes_for(account)]}}
    return df.to_dict("records"), columns, dropdown


def _filter_df(account, start: str | None, end: str | None) -> pd.DataFrame:
    df = account.transaction_data
    if start or end:
        date_str = df["date"].dt.strftime("%Y-%m-%d")
        if start:
            df = df[date_str >= start]
        if end:
            df = df[date_str <= end]
    return df


def _cross_frames(all_accounts, filtered_names: list, codes: list,
                  start: str | None, end: str | None,
                  str_filter: str | None = None,
                  float_mask: _FloatMask | None = None,
                  float_abs: bool = False) -> dict[str, pd.DataFrame]:
    """Return {acc_name: filtered_df} for accounts that have matching rows."""
    accounts = AccountsList([all_accounts[n] for n in filtered_names])

    if codes:
        account_frames: dict[str, list] = {}
        for code in codes:
            for acc_name, df in accounts.filter_by_code(
                code=code, start_date=start or "", end_date=end
            ).items():
                account_frames.setdefault(acc_name, []).append(df)
        frames = {name: pd.concat(fs) for name, fs in account_frames.items()}
    else:
        frames = {}
        for acc in accounts:
            df = _filter_df(acc, start, end)
            if not df.empty:
                frames[acc.name] = df

    if str_filter:
        result = {}
        for acc_name, df in frames.items():
            cols = _text_cols(df)
            if cols:
                mask = pd.concat([
                    df[c].astype(str).str.contains(str_filter, regex=True, case=False, na=False)
                    for c in cols
                ], axis=1).any(axis=1)
                df = df[mask]
            if not df.empty:
                result[acc_name] = df
        frames = result

    if float_mask is not None:
        result = {}
        for acc_name, df in frames.items():
            num_cols = [c for c in all_accounts[acc_name].numerical_names if c in df.columns]
            if num_cols:
                vals = df[num_cols].abs() if float_abs else df[num_cols]
                mask = vals.apply(float_mask).any(axis=1)
                df = df[mask]
            if not df.empty:
                result[acc_name] = df
        frames = result

    return frames


def _totals(account, rows: list) -> str:
    parts = [f"{col}: {sum(r.get(col, 0) or 0 for r in rows):+.2f}"
             for col in account.numerical_names]
    return "  |  ".join(parts)


def transactions_layout(all_accounts, account_options: list[dict],
                        default_account: str, lang: str = "en") -> html.Div:
    code_opts = [{"label": c, "value": c} for c in _all_codes(all_accounts)]
    return html.Div(
        [
            # ── button row (single-account mode) ──────────────────────────
            html.Div(
                id="txn-save-reload-row",
                children=[
                    dcc.Dropdown(id="table-account-selector",
                                 options=account_options, value=default_account,
                                 clearable=False, style={"width": "300px"}),
                    html.Button(t(lang, "save"), id="save-table-btn", n_clicks=0,
                                style={"marginLeft": "10px"}),
                    html.Button(t(lang, "reload_csvs"), id="reload-csv-btn", n_clicks=0,
                                style={"marginLeft": "10px"}),
                    html.Span(id="save-status",
                              style={"marginLeft": "10px", "color": "green"}),
                ],
                style=_SHOWN_ROW,
            ),
            # ── filter bar ────────────────────────────────────────────────
            html.Div(
                [
                    dcc.Input(id="txn-str-filter", type="text", debounce=True,
                              placeholder=t(lang, "ph_regex"),
                              style={"width": "240px", "fontFamily": "monospace"}),
                    dcc.Input(id="txn-float-filter", type="text", debounce=True,
                              placeholder=t(lang, "ph_amount"),
                              style={"width": "160px", "fontFamily": "monospace",
                                     "marginLeft": "8px"}),
                    dcc.Dropdown(id="txn-code-filter", options=code_opts, multi=True,
                                 placeholder=t(lang, "ph_filter_code"),
                                 style={"flex": "1", "marginLeft": "8px"}),
                ],
                style={"display": "flex", "alignItems": "center",
                       "marginBottom": "10px"},
            ),
            # ── single-account view ───────────────────────────────────────
            html.Div(
                id="txn-single",
                children=[
                    dash_table.DataTable(
                        id="transaction-table",
                        columns=[], data=[],
                        hidden_columns=["_idx"],
                        editable=True,
                        sort_action="native",
                        sort_mode="multi",
                        page_action="none",
                        fixed_rows={"headers": True},
                        style_table={"height": "calc(100vh - 300px)",
                                     "overflowY": "auto", "overflowX": "auto"},
                        style_cell=_TABLE_CELL,
                        style_header={"fontWeight": "bold"},
                        style_data_conditional=_NA_HIGHLIGHT,
                    ),
                    html.Div(id="transaction-totals", style=_TOTALS_STYLE),
                ],
            ),
            # ── cross-account view (hidden initially) ─────────────────────
            html.Div(
                id="txn-multi",
                style=_HIDDEN,
                children=[
                    html.Div(
                        [
                            # left: account list
                            html.Div(
                                [
                                    html.Div(t(lang, "accounts_heading"),
                                             style={"fontWeight": "bold",
                                                    "marginBottom": "8px",
                                                    "fontFamily": "monospace"}),
                                    dcc.RadioItems(
                                        id="txn-cross-selector",
                                        options=[],
                                        value=None,
                                        labelStyle={"display": "block",
                                                    "padding": "4px 6px",
                                                    "cursor": "pointer",
                                                    "fontFamily": "monospace"},
                                        inputStyle={"marginRight": "8px"},
                                    ),
                                ],
                                style={"width": "200px", "flexShrink": "0",
                                       "borderRight": "1px solid #e0e0e0",
                                       "paddingRight": "12px",
                                       "marginRight": "16px"},
                            ),
                            # right: table + save + totals
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Button("Save", id="txn-cross-save-btn",
                                                        n_clicks=0),
                                            html.Span("", id="txn-cross-save-status",
                                                      style={"marginLeft": "10px",
                                                             "color": "green",
                                                             "fontFamily": "monospace"}),
                                        ],
                                        style={"marginBottom": "8px"},
                                    ),
                                    dash_table.DataTable(
                                        id="txn-cross-table",
                                        columns=[], data=[],
                                        hidden_columns=["_idx"],
                                        editable=True,
                                        sort_action="native",
                                        sort_mode="multi",
                                        page_action="none",
                                        fixed_rows={"headers": True},
                                        style_table={"height": "calc(100vh - 300px)",
                                                     "overflowY": "auto",
                                                     "overflowX": "auto"},
                                        style_cell=_TABLE_CELL,
                                        style_header={"fontWeight": "bold"},
                                        style_data_conditional=_NA_HIGHLIGHT,
                                    ),
                                    html.Div("", id="txn-cross-totals",
                                             style=_TOTALS_STYLE),
                                ],
                                style={"flex": "1", "minWidth": "0"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "flex-start"},
                    ),
                ],
            ),
        ],
        style={"padding": "10px"},
    )


def register_transactions_callbacks(app, all_accounts) -> None:

    @app.callback(
        Output("table-account-selector", "options"),
        Output("table-account-selector", "value"),
        Input("filtered-accounts", "data"),
        State("table-account-selector", "value"),
    )
    def update_transactions_selector(filtered_names, current_value):
        options = [{"label": n, "value": n} for n in filtered_names]
        new_value = (
            current_value if current_value in filtered_names
            else (filtered_names[0] if filtered_names else None)
        )
        return options, new_value

    # ── main view-switching callback ──────────────────────────────────────────

    @app.callback(
        Output("transaction-table", "data"),
        Output("transaction-table", "columns"),
        Output("transaction-table", "dropdown"),
        Output("transaction-table", "editable"),
        Output("txn-cross-selector", "options"),
        Output("txn-cross-selector", "value"),
        Output("txn-single", "style"),
        Output("txn-multi", "style"),
        Output("txn-save-reload-row", "style"),
        Input("table-account-selector", "value"),
        Input("global-period", "data"),
        Input("txn-str-filter", "value"),
        Input("txn-float-filter", "value"),
        Input("txn-code-filter", "value"),
        State("filtered-accounts", "data"),
    )
    def update_txn_view(account_name, period_data, str_filter, float_str, codes, filtered_names):
        start = (period_data or {}).get("start")
        end = (period_data or {}).get("end")
        float_mask, float_abs = _parse_float_filter(float_str)

        if codes or str_filter or float_mask is not None:
            matched = _cross_frames(all_accounts, filtered_names or [], codes or [],
                                    start, end, str_filter, float_mask, float_abs)
            options = [{"label": n, "value": n} for n in matched]
            first = next(iter(matched), None)
            return ([], [], {}, False,
                    options, first,
                    _HIDDEN, {"display": "block"}, _HIDDEN)

        df = _filter_df(all_accounts[account_name], start, end)
        data, columns, dropdown = _fmt(all_accounts[account_name], df)
        return (data, columns, dropdown, True,
                [], None,
                {"display": "block"}, _HIDDEN, _SHOWN_ROW)

    # ── cross-account right panel ─────────────────────────────────────────────

    @app.callback(
        Output("txn-cross-table", "data"),
        Output("txn-cross-table", "columns"),
        Output("txn-cross-table", "dropdown"),
        Input("txn-cross-selector", "value"),
        Input("txn-str-filter", "value"),
        Input("txn-float-filter", "value"),
        Input("txn-code-filter", "value"),
        Input("global-period", "data"),
        State("filtered-accounts", "data"),
    )
    def update_cross_table(acc_name, str_filter, float_str, codes, period_data, filtered_names):
        if not acc_name:
            return [], [], {}
        start = (period_data or {}).get("start")
        end = (period_data or {}).get("end")
        float_mask, float_abs = _parse_float_filter(float_str)
        matched = _cross_frames(all_accounts, filtered_names or [], codes or [],
                                start, end, str_filter, float_mask, float_abs)
        if acc_name not in matched:
            return [], [], {}
        account = all_accounts[acc_name]
        data, columns, dropdown = _fmt(account, matched[acc_name])
        return data, columns, dropdown

    @app.callback(
        Output("txn-cross-totals", "children"),
        Input("txn-cross-table", "derived_virtual_data"),
        State("txn-cross-selector", "value"),
    )
    def update_cross_totals(rows, acc_name):
        if not rows or not acc_name:
            return ""
        return _totals(all_accounts[acc_name], rows)

    @app.callback(
        Output("txn-cross-save-status", "children"),
        Input("txn-cross-save-btn", "n_clicks"),
        State("txn-cross-table", "data"),
        State("txn-cross-selector", "value"),
        State("lang", "data"),
        prevent_initial_call=True,
    )
    def save_cross(_, rows, acc_name, lang):
        lang = lang or "en"
        if not acc_name:
            return ""
        account = all_accounts[acc_name]
        if isinstance(rows, dict):
            rows = [v for v in rows.values() if isinstance(v, dict) and "_idx" in v]
        n = 0
        for row in rows:
            idx, new_code = row["_idx"], row["code"]
            if account.transaction_data.loc[idx, "code"] != new_code:
                account.change_transaction_code(idx, new_code)
                n += 1
        if n:
            account.save()
        return t(lang, "saved_n", n=n) if n else t(lang, "nothing_to_save")

    # ── single-account callbacks ──────────────────────────────────────────────

    @app.callback(
        Output("save-status", "children"),
        Output("transaction-table", "data", allow_duplicate=True),
        Output("transaction-table", "columns", allow_duplicate=True),
        Output("transaction-table", "dropdown", allow_duplicate=True),
        Input("save-table-btn", "n_clicks"),
        State("transaction-table", "data"),
        State("table-account-selector", "value"),
        State("global-period", "data"),
        State("lang", "data"),
        prevent_initial_call=True,
    )
    def save_table(_, rows, account_name, period_data, lang):
        lang = lang or "en"
        account = all_accounts[account_name]
        if isinstance(rows, dict):
            rows = [v for v in rows.values() if isinstance(v, dict) and "_idx" in v]
        n = 0
        for row in rows:
            idx, new_code = row["_idx"], row["code"]
            if account.transaction_data.loc[idx, "code"] != new_code:
                account.change_transaction_code(idx, new_code)
                n += 1
        if n:
            account.save()
        start = (period_data or {}).get("start")
        end = (period_data or {}).get("end")
        data, columns, dropdown = _fmt(account, _filter_df(account, start, end))
        return (t(lang, "saved_n", n=n) if n else t(lang, "nothing_to_save"),
                data, columns, dropdown)

    @app.callback(
        Output("save-status", "children", allow_duplicate=True),
        Output("transaction-table", "data", allow_duplicate=True),
        Output("transaction-table", "columns", allow_duplicate=True),
        Output("transaction-table", "dropdown", allow_duplicate=True),
        Input("reload-csv-btn", "n_clicks"),
        State("table-account-selector", "value"),
        State("global-period", "data"),
        State("lang", "data"),
        prevent_initial_call=True,
    )
    def reload_csvs(_, account_name, period_data, lang):
        lang = lang or "en"
        account = all_accounts[account_name]
        account.force_csv_reload()
        account.save()
        start = (period_data or {}).get("start")
        end = (period_data or {}).get("end")
        data, columns, dropdown = _fmt(account, _filter_df(account, start, end))
        return t(lang, "reloaded"), data, columns, dropdown

    @app.callback(
        Output("transaction-totals", "children"),
        Input("transaction-table", "derived_virtual_data"),
        State("table-account-selector", "value"),
    )
    def update_single_totals(rows, account_name):
        if not rows or not account_name:
            return ""
        return _totals(all_accounts[account_name], rows)
