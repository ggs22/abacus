import argparse

from Accounts import *
from View import View


def quit_abacus():
    exit(0)


def parse_arguments():
    # TODO remove arguments
    parser = argparse.ArgumentParser(description='Abacus')
    parser.add_argument('-i', '--desjardins_input_path', type=str, help='Choose the input csv files path',
                        required=False, default='./desjardins_csv_files')
    parser.add_argument('-p', '--desjardins_ppcard_input_path', type=str, help='Choose the input csv files path',
                        required=False, default='./desjardins_ppcard_csv_files')
    parser.add_argument('-c', '--capital_one_input_path', type=str, help='Choose the input csv files path',
                        required=False, default='./capital_one_csv_files')

    return parser.parse_args()


def start_gui():
    view = View()

    kwargs = {'year': 2021}
    f = desjardins_op.barplot(**kwargs)
    view.inscribe_dataframe(desjardins_op.get_data(**kwargs))
    view.display_figure(fig=f)

    view.start()


if __name__ == '__main__':

    args = parse_arguments()

    start_gui()
