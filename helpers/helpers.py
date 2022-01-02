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
    daily_hours = 7.5
    hourly_rate = 35.9
    clear_pay_rate = 0.6887

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


def calculate_pay(days: int, clear=True):
    daily_hours = 7.5
    hourly_rate = 35.9
    rate = 0.6887 if clear else 1
    return round(days * daily_hours * hourly_rate * rate, 2)
