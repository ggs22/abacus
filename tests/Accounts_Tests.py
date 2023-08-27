import matplotlib.pyplot as plt
import argparse

from accounting import accounts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_new',
                        help="Wether or not we want to force a new Monte-Carlo simulation computation.",
                        action='store_true')
    parser.add_argument('--avg_interval',
                        help="The number of passed days over which spending averages, standard deviations and daily "
                             "frequencies are calculated.",
                        default=365,
                        type=int)
    parser.add_argument('--montecarlo_iterations',
                        help="The number of iterations run for Monte-Carlo-simulated spendings.",
                        default=200,
                        type=int)

    return parser.parse_args()


if __name__ == "__main__":
    "tests"

    args = parse_args()

    # accounts.get_prediction()
    for acc in accounts:
        for m in range(0, 3):
            acc.barplot(period_seed_date=f'2022-{8-m}')
    plt.show()

    # accounts.plot_balance_prediction()
