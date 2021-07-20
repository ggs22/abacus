import argparse
import time

import colorama
import re
import threading
import queue

from CSVParser import CSVParser
from Grapher import Grapher
from View import View


def quit_abacus():
    exit(0)


class Command:
    def __init__(self, flags: list, help_msg: str, name: str, method, **kwargs):
        if type(flags) is not list:
            raise ValueError('Expected list')
        self.flags = flags
        self.help_msg = help_msg
        self.method = method
        self.params = kwargs
        self.name = name

    def execute(self):
        self.method(**self.params)

    def print_flag_and_name(self):
        print(f'{colorama.Fore.LIGHTBLUE_EX}'
              f'{self.flags[0]}: '
              f'{" " * (6 - len(self.flags[0]))}'
              f'{colorama.Fore.RESET}'
              f'{self.name}')


def process_command(user_command):
    for command in commnands:
        r = re.match('(pcom)( -m )([0-9]{1,2})', user_command)
        if r is not None:
            command.params = {'month': int(r[3])}
            user_command = r[1]
        if user_command in command.flags:
            command.execute()
            break


def print_commands():
    for command in commnands:
        command.print_flag_and_name()


def print_menu():
    # print('\033[1J')  # clear screan
    # print('\033[H')  # cursor at top left corner
    if not init:
        print(f'\033[{12 + len(commnands)}A')  # Cursor up # times in [#A
    print(colorama.Fore.LIGHTGREEN_EX)
    print('=========================================\n'
          '   ##   #####    ##    ####  ##   #  ####\n'
          ' ##  #  ##   #  ## #  ##   # ##   # ##\n'
          '##    # #####  ##   # ##     ##   #  ####\n'
          '####### ##   # ###### ##     ##   #      #\n'
          '##    # ##   # ##   # ##   # ##   # ##   #\n'
          '##    # #####  ##   #  ####   ####   ####\n'
          '=========================================')
    print(colorama.Fore.RESET)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Abacus')
    parser.add_argument('-i', '--desjardins_input_path', type=str, help='Choose the input csv files path',
                        required=False, default='./desjardins_csv_files')
    parser.add_argument('-p', '--desjardins_ppcard_input_path', type=str, help='Choose the input csv files path',
                        required=False, default='./desjardins_ppcard_pdf_files')
    parser.add_argument('-c', '--capital_one_input_path', type=str, help='Choose the input csv files path',
                        required=False, default='./capital_one_csv_files')

    return parser.parse_args()


def start_gui():
    view = View(csv_parser)

    view.set_desjardons_mc_treeview()
    view.display_desjardins_op()
    view.display_desjardins_op2()
    view.display_desjardins_op3()

    view.start()


if __name__ == '__main__':

    th = threading.Thread(target=start_gui, daemon=True)

    args = parse_arguments()
    init = True
    # q = queue.Queue()

    csv_parser = CSVParser(desjardins_input_path=args.desjardins_input_path,
                           desjardins_ppcard_input_path=args.desjardins_ppcard_input_path,
                           capital_one_input_path=args.capital_one_input_path)
    grapher = Grapher(accounts=csv_parser.accounts)
    desjardins_op, desjardins_mc, desjardins_sloan, capital_one, ppcard = csv_parser.get_accounts()

    commnands = list()

    commnands.append(Command(['ty'], help_msg='Plot Total Yearly Balance',
                             name='Total Yearly Balance', method=grapher.plot_year_total, year=2020))

    commnands.append(Command(['udd'], help_msg='Update Desjardins data from new csv files',
                             name='Update Desjardins data', method=csv_parser.update_desjardins_data))
    commnands.append(Command(['uco'], help_msg='Update Capital One data from csv files',
                             name='Update Capital One data', method=csv_parser.update_capital_one_data,
                             input_path='./capital_one_csv_files'))
    commnands.append(Command(['pdy'], help_msg='Plot Desjardins Yearly Balance',
                             name='Desjardins Yearly Balance', method=grapher.plot_year_desjardins, year=2020))
    commnands.append(Command(['pcoy'], help_msg='Plot Capital One Yearly Expenses',
                             name='Capital One Yearly Expenses', method=grapher.plot_year_capital_one, year=2020))
    commnands.append(Command(['pcoa'], help_msg='Plot Capital One Monthly Expenses',
                             name='Capital One Monthly Expenses', method=grapher.plot_all_months_capital_one))
    commnands.append(Command(['pcom'], help_msg='Plot Capital One month balance',
                             name='Capital One Monthly balance', method=grapher.plot_month_capital_one, month=10))
    commnands.append(Command(['pmc'], help_msg='Plot Desjardins MC balance',
                             name='Plot Desjardins MC balance', method=grapher.plot_desjardins_mc))
    commnands.append(Command(['pop'], help_msg='Plot Desjardins OP balance',
                             name='Plot Desjrdins OP balance', method=grapher.plot_desjardins_op))
    commnands.append(Command(['pco'], help_msg='Plot Capital One balance',
                             name='Plot Capital One balance', method=grapher.plot_capital_one))
    commnands.append(Command(['s'], help_msg='Save current codes',
                             name='save', method=csv_parser.save_all_accounts))
    commnands.append(Command(['q'], help_msg='Exit the program',
                             name='quit', method=quit_abacus))

    # start gui thread
    th.start()

    while True:
        print_menu()
        print_commands()
        res = input(' ' * 80 + '\b' * 80 + '>> ')
        process_command(res)
        init = False

        time.sleep(0.2)
