#%%
import os
import colorama
import re
import pyperclip


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
