import argparse
import accounting as acc

import datetime
from accounting import accounts, desjardins_mc
from accounting.Account import Account


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


def plot_years():
    for year in [2019, 2020, 2021]:
        accounts.barpdlot(year=year, show=True)


def print_accounts_info():
    for acc in accounts:
        print(acc.metadata.name.name)
        print(acc.most_recent_date)
        print(acc.get_current_balance())


def plot_predictions(force_new: bool = False, avg_interval=365, montecarlo_iterations=200):
    start_date = datetime.date(year=2020, month=1, day=1)
    sim_dates = list()
    # sim_dates += [datetime.date(year=2023, month=1, day=31)]
    sim_dates += [datetime.date(year=2023, month=5, day=15)]
    sim_dates += [datetime.date(year=2023, month=5, day=31)]
    # for i in range(3, 0, -1):
    #     sim_dates += [datetime.date(year=2022, month=(12 - i), day=6)]
    end_date = datetime.date(year=2023, month=12, day=31)

    acc.desjardins_mc.plot_prediction_compare(get_avg_method=accounts.get_data_range_daily_average,
                                              start_date=start_date,
                                              sim_dates=sim_dates,
                                              end_date=end_date,
                                              show=True,
                                              force_new=force_new,
                                              avg_interval=avg_interval,
                                              montecarl_iterations=montecarlo_iterations)


def get_averages():
    avg = accounts.get_daily_average(year=2021)
    print(avg)
    avg = accounts.get_daily_average()
    print(avg)


def bp_years(years):
    for y in years:
        accounts.barplot(year=y, show=True)


def bp_last_months(num_months: int = 3):
    d = datetime.datetime.today().date()
    mi = d.month
    y = d.year
    count = 0
    for i in range(0, num_months):
        m = (mi - count) % 12
        m = m * (m > 0) + 12 * (m == 0)
        if m == 12 and mi < 12:
            y = (y - 1) * (m == 12) + y * (m < 12)
        accounts.barplot(year=y, month=m, show=True)
        count += 1


def bp_current_month():
    accounts.barplot(year=datetime.datetime.today().year,
                     month=datetime.datetime.today().month)


if __name__ == "__main__":
    "tests"

    args = parse_args()

    # for c in desjardins_accounts:
    #     c.clear_month(year=2023, month=7, inplace=True)
    #     c.update_from_raw_files()
    # pred = desjardins_mc.get_predicted_balance(end_date=datetime.date(year=2023, month=12, day=31),
    #                                            force_new=True)
    # bp_last_months(num_months=6)

    # cProfile.run(statement='plot_predictions(force_new=False)', sort='cumtime')
    # plot_predictions(force_new=args.force_new,
    #                  avg_interval=args.avg_interval,
    #                  montecarlo_iterations=args.montecarlo_iterations)
    # print(accounting.get_most_recent_transaction_date())
    # bp_last_months(3)
    # yearly_summary = accounting.get_yearly_summary(year=2022)
    # accounting.plot_yearly_summary(year=2022, columns=yearly_summary.index[yearly_summary.index != 'pay'])