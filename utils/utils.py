#%%
import os
import colorama
import re
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


def get_project_root():
    return os.path.dirname(os.path.dirname(__file__))


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
    switch_week = False
    pay_day_counts = [0, 0]

    # LTI pays are max 16 days after pay period (fix dates pays)
    end_date = end_date + datetime.timedelta(days=16)

    for i in range(0, (end_date-start_date).days):
        d = start_date + datetime.timedelta(days=i)
        if d.weekday() in [0, 1, 2, 3, 4]:
            pay_day_counts[switch_week] += 1
        if d.day == 15:
            pay_amount = calculate_pay(pay_day_counts[not switch_week])
            pay_day_counts[not switch_week] = 0
            print(f'pay,{d.year},{d.month},{d.day},0,{pay_amount},pay')
            switch_week = not switch_week
        elif d.day == 1:
            tmp = d - datetime.timedelta(days=1)
            pay_amount = calculate_pay(pay_day_counts[not switch_week])
            pay_day_counts[not switch_week] = 0
            print(f'pay,{tmp.year},{tmp.month},{tmp.day},0,{pay_amount},pay')
            switch_week = not switch_week


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


def calculate_pay(days: int, clear=True):
    """
    Calculates the pay for a given number of work days
    :param days: number of work days
    :param clear: if true, taxes will be deducted from result
    :return: pay amount
    """

    daily_hours = 7.5
    hourly_rate = (85000/1950)
    rate = 0.6887 if clear else 1
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

if __name__ == "__main__":
    sdate = datetime.date(year=2022, month=11, day=16)
    edate = datetime.date(year=2023, month=11, day=15)

    print_lti_pays(start_date=sdate, end_date=edate)