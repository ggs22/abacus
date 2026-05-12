import calendar
import re as _re

_YEAR_RE    = _re.compile(r'^(\d{4}|\d{2})$')
_QUARTER_RE = _re.compile(r'^(\d{4}|\d{2})[qQ]([1-4])$')
_MONTH_RE   = _re.compile(r'^(\d{4}|\d{2})-(\d{2})$')
_DAY_RE     = _re.compile(r'^(\d{4}|\d{2})-(\d{2})-(\d{2})$')
_Q_ABBREV   = _re.compile(r'^[qQ]([1-4])$')
_TWO_DIGITS = _re.compile(r'^\d{2}$')

_QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}


def _expand_year(y_str: str) -> int:
    y = int(y_str)
    return 2000 + y if len(y_str) == 2 else y


def _parse_token(token: str) -> dict:
    m = _DAY_RE.match(token)
    if m:
        y_str, mo_str, d_str = m.groups()
        y, mo, d = _expand_year(y_str), int(mo_str), int(d_str)
        s = f"{y:04d}-{mo:02d}-{d:02d}"
        return {"gran": "day", "yd": len(y_str), "start": s, "end": s, "year": y, "month": mo}

    m = _MONTH_RE.match(token)
    if m:
        y_str, mo_str = m.groups()
        y, mo = _expand_year(y_str), int(mo_str)
        last = calendar.monthrange(y, mo)[1]
        return {"gran": "month", "yd": len(y_str),
                "start": f"{y:04d}-{mo:02d}-01", "end": f"{y:04d}-{mo:02d}-{last:02d}",
                "year": y, "month": mo}

    m = _QUARTER_RE.match(token)
    if m:
        y_str, q_str = m.groups()
        y, q = _expand_year(y_str), int(q_str)
        s_mo, e_mo = _QUARTER_MONTHS[q]
        last = calendar.monthrange(y, e_mo)[1]
        return {"gran": "quarter", "yd": len(y_str),
                "start": f"{y:04d}-{s_mo:02d}-01", "end": f"{y:04d}-{e_mo:02d}-{last:02d}",
                "year": y, "month": None}

    m = _YEAR_RE.match(token)
    if m:
        y_str = m.group(1)
        y = _expand_year(y_str)
        return {"gran": "year", "yd": len(y_str),
                "start": f"{y:04d}-01-01", "end": f"{y:04d}-12-31",
                "year": y, "month": None}

    raise ValueError(f"Unrecognised date token: {token!r}")


def _parse_right_end(token: str, left: dict) -> str:
    gran, yd = left["gran"], left["yd"]

    if gran == "quarter":
        m = _Q_ABBREV.match(token)
        if m:
            q = int(m.group(1))
            _, e_mo = _QUARTER_MONTHS[q]
            last = calendar.monthrange(left["year"], e_mo)[1]
            return f"{left['year']:04d}-{e_mo:02d}-{last:02d}"

    if gran == "month" and _TWO_DIGITS.match(token):
        mo = int(token)
        last = calendar.monthrange(left["year"], mo)[1]
        return f"{left['year']:04d}-{mo:02d}-{last:02d}"

    if gran == "day" and _TWO_DIGITS.match(token):
        d = int(token)
        return f"{left['year']:04d}-{left['month']:02d}-{d:02d}"

    right = _parse_token(token)
    if right["gran"] != gran:
        raise ValueError(
            f"Range sides must have the same granularity (left={gran}, right={right['gran']})"
        )
    if right["yd"] != yd:
        raise ValueError(
            f"Range sides must use the same year representation "
            f"({yd}-digit vs {right['yd']}-digit)"
        )
    return right["end"]


def parse_period(s: str) -> tuple[str, str]:
    """Parse a period string into (start_date, end_date) ISO strings.

    Accepts: year (2026), year-month (2026-01), quarter (2026Q1),
             day (2026-01-15), or a range with ':' (2026-01:2026-03).
    Two-digit years are expanded to 2000+.
    """
    s = s.strip()
    if ":" in s:
        left_str, right_str = s.split(":", 1)
        left = _parse_token(left_str.strip())
        end = _parse_right_end(right_str.strip(), left)
        return left["start"], end
    parsed = _parse_token(s)
    return parsed["start"], parsed["end"]
