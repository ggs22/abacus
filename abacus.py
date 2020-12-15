import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import argparse

from csv_parser import CSV_Parser

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

    csv_parser = CSV_Parser(args.desjardins_input_path, args.capital_one_input_path)
    a, b, c, d = csv_parser.get_data()

    a_mars, b_mars, c_mars, d_mars = csv_parser.get_data_by_date(month=3);

    print(a)

    input('Press any key to quit...')