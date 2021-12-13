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


def print_pays(start_date: datetime.date, end_date: datetime.date):
    pay_day_count = 0
    for i in range(0, (end_date-start_date).days):
        d = start_date + datetime.timedelta(days=i)
        if d.weekday() in [0, 1, 2, 3, 4]:
            pay_day_count += 1
        if d.day == 15:
            pay_amount = round(pay_day_count * 7.5 * 35.88 * 0.8, 2)
            pay_day_count = 0
            print(f'pay,{d.year},{d.month},{d.day},0,{pay_amount},pay')
        else:
            if d.month in [1, 3, 5, 7, 10, 12]:
                if d.day == 31:
                    pay_amount = round(pay_day_count * 7.5 * 35.88 * 0.8, 2)
                    pay_day_count = 0
                    print(f'pay,{d.year},{d.month},{d.day},0,{pay_amount},pay')
            if d.month in [4, 6, 8, 9, 11]:
                if d.day == 30:
                    pay_amount = round(pay_day_count * 7.5 * 35.88 * 0.8, 2)
                    pay_day_count = 0
                    print(f'pay,{d.year},{d.month},{d.day},0,{pay_amount},pay')
            if d.month in [2]:
                if d.day == 28:
                    pay_amount = round(pay_day_count * 7.5 * 35.88 * 0.8, 2)
                    pay_day_count = 0
                    print(f'pay,{d.year},{d.month},{d.day},0,{pay_amount},pay')
