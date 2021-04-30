import argparse
import colorama
import re

from CSV_Parser import CSV_Parser
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
                        required=False)
    parser.add_argument('-c', '--capital_one_input_path', type=str, help='Choose the input csv files path',
                        required=False)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    init = True

    csv_parser = CSV_Parser(args.desjardins_input_path, args.capital_one_input_path)
    grapher = Grapher(parser=csv_parser)
    desjardins_op, desjardins_mc, desjardins_sloan, capital_one = csv_parser.get_data()

    commnands = list()

    commnands.append(Command(['ty'], help_msg='Plot Total Yearly Balance',
                             name='Total Yearly Balance', method=grapher.plot_year_total, year=2020))

    commnands.append(Command(['cco'], help_msg='Update Capital One data from clipboard',
                             name='Update Capital One data - clipboard',
                             method=csv_parser.update_desjardins_data_from_clipboard))
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
                             name='save', method=quit_abacus))
    commnands.append(Command(['q'], help_msg='Exit the program',
                             name='quit', method=quit_abacus))

    view = View()
    view.start()

    # while True:
    #
    #     print_menu()
    #     print_commands()
    #     res = input(' ' * 80 + '\b' * 80 + '>> ')
    #     process_command(res)
    #     init = False
