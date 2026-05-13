import json
from pathlib import Path

from dash import html, dcc, Input, Output, State, ctx

from .i18n import t

_PROFILES_PATH = Path(__file__).parent.parent / "accounting" / "accounts" / "profiles.json"


def _load() -> dict:
    if _PROFILES_PATH.exists():
        with open(_PROFILES_PATH) as f:
            return json.load(f)
    return {}


def _save(profiles: dict) -> None:
    with open(_PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False)


def profiles_layout(lang: str = "en") -> html.Div:
    options = [{"label": k, "value": k} for k in _load()]
    return html.Div(
        [
            html.Label(t(lang, "profile"),
                       style={"fontWeight": "bold", "marginRight": "8px",
                              "flexShrink": "0", "fontFamily": "monospace"}),
            dcc.Dropdown(
                id="profile-selector",
                options=options,
                value=None,
                clearable=True,
                placeholder=t(lang, "ph_load_profile"),
                style={"width": "200px"},
            ),
            dcc.Input(
                id="profile-name-input",
                placeholder=t(lang, "ph_new_profile"),
                debounce=False,
                style={"marginLeft": "8px", "width": "200px", "fontFamily": "monospace"},
            ),
            html.Button(t(lang, "save"), id="profile-save-btn", n_clicks=0,
                        style={"marginLeft": "8px"}),
            html.Button(t(lang, "delete"), id="profile-delete-btn", n_clicks=0,
                        style={"marginLeft": "6px", "color": "red"}),
            html.Span("", id="profile-status",
                      style={"marginLeft": "10px", "color": "green",
                             "fontFamily": "monospace"}),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "padding": "6px 10px",
            "borderBottom": "1px solid var(--border)",
            "backgroundColor": "var(--bg2)",
        },
    )


def register_profiles_callbacks(app) -> None:

    @app.callback(
        Output("category-filter", "value"),
        Output("family-filter", "value"),
        Output("institution-filter", "value"),
        Output("global-period-input", "value"),
        Input("profile-selector", "value"),
        prevent_initial_call=True,
    )
    def load_profile(name):
        if not name:
            return None, None, None, "::"
        p = _load().get(name, {})
        return p.get("categories"), p.get("families"), p.get("institutions"), p.get("period", "::")

    @app.callback(
        Output("profile-selector", "options"),
        Output("profile-selector", "value", allow_duplicate=True),
        Output("profile-name-input", "value"),
        Output("profile-status", "children"),
        Input("profile-save-btn", "n_clicks"),
        Input("profile-delete-btn", "n_clicks"),
        State("profile-name-input", "value"),
        State("profile-selector", "value"),
        State("category-filter", "value"),
        State("family-filter", "value"),
        State("institution-filter", "value"),
        State("global-period-input", "value"),
        State("lang", "data"),
        prevent_initial_call=True,
    )
    def save_or_delete(_, __, name_input, selected, categories, families, institutions, period, lang):
        lang = lang or "en"
        profiles = _load()
        triggered = ctx.triggered_id

        if triggered == "profile-save-btn":
            name = (name_input or "").strip()
            if not name:
                options = [{"label": k, "value": k} for k in profiles]
                return options, selected, name_input, t(lang, "profile_name_required")
            profiles[name] = {
                "categories": categories or [],
                "families": families or [],
                "institutions": institutions or [],
                "period": period or "::",
            }
            _save(profiles)
            options = [{"label": k, "value": k} for k in profiles]
            return options, name, "", t(lang, "profile_saved", name=name)

        if triggered == "profile-delete-btn":
            if not selected or selected not in profiles:
                options = [{"label": k, "value": k} for k in profiles]
                return options, selected, name_input, t(lang, "profile_nothing_to_delete")
            del profiles[selected]
            _save(profiles)
            options = [{"label": k, "value": k} for k in profiles]
            return options, None, name_input, t(lang, "profile_deleted", name=selected)

        options = [{"label": k, "value": k} for k in profiles]
        return options, selected, name_input, ""
