import os


def print_step(stp_name: str, stp_ix: int, stp_tot: int, msg: str = ''):
    print('######################################################################')
    print(f'{stp_name} ({stp_ix}/{stp_tot})' + f' - ' * (msg != '') + msg + '\n')


def get_project_root():
    return os.path.dirname(os.path.dirname(__file__))
