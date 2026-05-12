import json
from pathlib import Path

from dash import html, dcc, Input, Output, State, ctx

_PROFILES_PATH = Path(__file__).parent.parent / "accounting" / "accounts" / "profiles.json"


def _load() -> dict:
    if _PROFILES_PATH.exists():
        with open(_PROFILES_PATH) as f:
            return json.load(f)
    return {}


def _save(profiles: dict) -> None:
    with open(_PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False)


def profiles_layout() -> html.Div:
    options = [{"label": k, "value": k} for k in _load()]
    return html.Div(
        [
            html.Label("Profile", style={"fontWeight": "bold", "marginRight": "8px",
                                         "flexShrink": "0", "fontFamily": "monospace"}),
            dcc.Dropdown(
                id="profile-selector",
                options=options,
                value=None,
                clearable=True,
                placeholder="Load profile…",
                style={"width": "200px"},
            ),
            dcc.Input(
                id="profile-name-input",
                placeholder="New profile name…",
                debounce=False,
                style={"marginLeft": "8px", "width": "200px", "fontFamily": "monospace"},
            ),
            html.Button("Save", id="profile-save-btn", n_clicks=0,
                        style={"marginLeft": "8px"}),
            html.Button("Delete", id="profile-delete-btn", n_clicks=0,
                        style={"marginLeft": "6px", "color": "red"}),
            html.Span("", id="profile-status",
                      style={"marginLeft": "10px", "color": "green",
                             "fontFamily": "monospace"}),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "padding": "6px 10px",
            "borderBottom": "1px solid #ddd",
            "backgroundColor": "#f8f8f8",
        },
    )


def register_profiles_callbacks(app) -> None:

    @app.callback(
        Output("category-filter", "value"),
        Output("family-filter", "value"),
        Output("institution-filter", "value"),
        Input("profile-selector", "value"),
        prevent_initial_call=True,
    )
    def load_profile(name):
        if not name:
            return None, None, None
        p = _load().get(name, {})
        return p.get("categories"), p.get("families"), p.get("institutions")

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
        prevent_initial_call=True,
    )
    def save_or_delete(_, __, name_input, selected, categories, families, institutions):
        profiles = _load()
        triggered = ctx.triggered_id

        if triggered == "profile-save-btn":
            name = (name_input or "").strip()
            if not name:
                options = [{"label": k, "value": k} for k in profiles]
                return options, selected, name_input, "Name required."
            profiles[name] = {
                "categories": categories or [],
                "families": families or [],
                "institutions": institutions or [],
            }
            _save(profiles)
            options = [{"label": k, "value": k} for k in profiles]
            return options, name, "", f"Saved '{name}'."

        if triggered == "profile-delete-btn":
            if not selected or selected not in profiles:
                options = [{"label": k, "value": k} for k in profiles]
                return options, selected, name_input, "Nothing to delete."
            del profiles[selected]
            _save(profiles)
            options = [{"label": k, "value": k} for k in profiles]
            return options, None, name_input, f"Deleted '{selected}'."

        options = [{"label": k, "value": k} for k in profiles]
        return options, selected, name_input, ""
