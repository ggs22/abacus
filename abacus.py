import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from csv_parser import CSV_Parser
from csv_parser import CashFLow

def parse_arguments():
    parser = argparse.ArgumentParser(description='Abacus')
    parser.add_argument('-i', '--desjardins_input_path', type=str, help='Choose the input csv files path',
                        required=False)
    parser.add_argument('-c', '--capital_one_input_path', type=str, help='Choose the input csv files path',
                        required=False)
    parser.add_argument('-w', '--weight_path', type=str, help='Choose the pth file path', required=False)
    parser.add_argument('--threshold', type=float, help='Choose the pth file path', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    a, b, c, d = CSV_Parser(args.desjardins_input_path, args.capital_one_input_path).get_data()

    a.get_data_by_date()

    print(a)