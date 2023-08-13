import datetime
import re

from typing import Tuple, Union


def get_period_tuple(date_str: str) -> Tuple[datetime.date, str]:
    """returns date string, date format and period unit (day, month or year)"""

    period_tuple = None
    year_pattern = re.compile(pattern="(\d{4})$")
    month_pattern = re.compile(pattern="(\d{4})([ _/-])(\d{1,2})$")
    day_pattern = re.compile(pattern="(\d{4})([ _/-])(\d{2})([ _/-])(\d{1,2})$")
    m = day_pattern.match(date_str)
    if m:
        period_tuple = datetime.datetime.strptime(f"{m[1]}-{m[3]}-{m[5]}", "%Y-%m-%d").date(), 'day'
    m = month_pattern.match(date_str)
    if m:
        period_tuple = datetime.datetime.strptime(f"{m[1]}-{m[3]}", "%Y-%m").date(), 'month'
    m = year_pattern.match(date_str)
    if m:
        period_tuple = datetime.datetime.strptime(f"{m[1]}", "%Y").date(), 'year'
    if period_tuple is None:
        raise RuntimeError(f"Invalide date format: {date_str}. Accepted formats are: "
                           f"YYYY, YYYY-MM, YYYY-MM-DD.")

    return period_tuple


def get_last_day_of_the_period(seed_date: datetime.date, period_unit: str) -> datetime.date:
    last_date = None
    if period_unit == 'day':
        last_date = seed_date
    if period_unit == 'month':
        last_date = datetime.date(seed_date.year, seed_date.month + 1, 1) - datetime.timedelta(days=1)
    if period_unit == 'year':
        last_date = datetime.date(seed_date.year + 1, 1, 1) - datetime.timedelta(days=1)
    if last_date is None:
        raise ValueError(f"Period muse be 0 (yyyy-mm-dd), 1 (yyyy-mm) or 2 (yyyy). Got {period_unit}.")
    return last_date


def get_period_bounds(seed_date: str, end_date: str = "") -> Tuple[datetime.date, Union[datetime.date, None], int]:
    """
    Returns a period defined by a start and end date, and the number of days of that period. It supports the following
    date formats: YYYY, YYYY-MM, YYYY-MM-DD. The period is defined by the smallest time unit among YYYY, MM and DD.
    For example:
        '2020-02', '2020-05' would return a period of 4 months from 2020-02-01 to 2020-05-31 of 120 days.
        '2019', '2020-05' would return a period of 17 months from 2019-01-01 to 2020-05-31 of 516 days.
    """
    first_day, period_type = get_period_tuple(seed_date)

    if end_date != "":
        end_tuple = get_period_tuple(end_date)
    else:
        end_tuple = (first_day, period_type)

    last_day = get_last_day_of_the_period(*end_tuple)

    return first_day, last_day


months_map = {
    1: ["january", 31],
    2: ["february", 28],
    3: ["march", 31],
    4: ["april", 30],
    5: ["may", 31],
    6: ["june", 30],
    7: ["july", 31],
    8: ["august", 31],
    9: ["september", 30],
    10: ["october", 31],
    11: ["november", 30],
    12: ["december", 31]
}


def print_pay_dates(start_date: datetime.date = datetime.datetime.today().date(),
                    end_date: datetime.date = datetime.datetime.today().date() + datetime.timedelta(days=90)):
    date = start_date
    for i in range (0, (end_date - start_date).days // 7):
        date = date + datetime.timedelta(days=i*14)
        if date > end_date:
            break
        # Monday == 1 ... Sunday == 7
        if date.isoweekday() == 4:
            print(f'pay,{date.year},{date.month},{date.day},0,1200,pay')