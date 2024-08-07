#%%
import os
import colorama
import re

import numpy as np
import pandas as pd
import pyperclip
import datetime


def print_step(stp_name: str, stp_ix: int, stp_tot: int, msg: str = ''):
    print(f'{colorama.Fore.BLUE}'
          f'######################################################################'
          f'{colorama.Fore.RESET}')

    print(f'{stp_name}'
          f'{colorama.Fore.YELLOW}'
          f' ({stp_ix}/{stp_tot})'
          f'{colorama.Fore.RESET}'
          f' - ' * (msg != '') + msg + '\n')


def get_css_selector_from_html(html_code: str, wait_time=5):
    pattern = "<(.*?) (.*?)>"
    regex = re.compile(pattern=pattern)
    found = regex.findall(html_code)[0]

    pattern = " ?(.*?)=\"(.*?)\""
    regex = re.compile(pattern)
    bracketed_single_quote = regex.sub(repl=r"[\1='\2']", string=found[1])
    css_selector = f"\"{found[0]}{bracketed_single_quote}\""
    ret = "WebDriverWait(driver, " + \
          str(wait_time) + \
          ").until(ec.presence_of_element_located((By.CSS_SELECTOR, " +\
          css_selector.replace("\'", "'") + \
          ")))"

    pyperclip.copy(ret)
    print(ret)


def print_lti_pays(start_date: datetime.date, end_date: datetime.date):
    """
    :param start_date: first day of pay period
    :param end_date: last day of pay period
    :return: prints pay tupples for the "desjardins_planned_transactions.csv" file
    """
    last_count = 0
    current_count = 0

    # LTI pays are max 16 days after pay period (fix dates pays)
    end_date = end_date + datetime.timedelta(days=16)

    def _format_timestamp(year, month, date, amount):
        return f"[{amount}, \"{year}-{month:02}-{date:02}\", \"pay\"],"

    for i in range(0, (end_date-start_date).days):
        d = start_date + datetime.timedelta(days=i)
        is_business_day = d.weekday() in [0, 1, 2, 3, 4]
        if is_business_day:
            current_count += 1
        if d.day == 15:
            pay_amount = calculate_pay(last_count)
            last_count = current_count
            current_count = 0
            print(_format_timestamp(d.year, d.month, d.day, pay_amount))
        elif d.day == 1:
            payday = d - datetime.timedelta(days=1)  # we get the last day of the preceding month
            pay_amount = calculate_pay(last_count)
            b_day_shift = 1 if d.weekday() in [0, 1, 2, 3, 4] else 0
            last_count = current_count - b_day_shift
            current_count = b_day_shift
            print(_format_timestamp(d.year, d.month, d.day, pay_amount))
            # switch_week = not switch_week


def calculate_pay(days: int, clear=True):
    """
    Calculates the pay for a given number of work days
    :param days: number of work days
    :param clear: if true, taxes will be deducted from result
    :return: pay amount
    """

    daily_hours = 7.5
    hourly_rate = np.round(96000/1950, 2)
    rate = 0.671 if clear else 1
    return round(days * daily_hours * hourly_rate * rate, 2)


def calculate_mc_interest(amount:float, days: int):
    """
    Calculates the accumulated interest per day for desjardins MC account
    :param days: number of days for which the interest is capitalized
    :return: interest amount
    """
    interest_rate = 0.0295 / 365.24
    return amount * (1+interest_rate) ** days


root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, 'data')
pickle_dir = os.path.join(root_dir, 'pickle_objects')


if __name__ == "__main__":
    sdate = datetime.date(year=2023, month=12, day=1)
    edate = datetime.date(year=2026, month=12, day=31)

    print_lti_pays(start_date=sdate, end_date=edate)


def mad(x: pd.Series) -> pd.Series:
    med = x.median()
    res = abs(x - med)
    mad = res.median()
    return mad