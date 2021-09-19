import os
import colorama


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
