from Accounts import *


def plot_years():
    for year in [2019, 2020, 2021]:
        accounts.barplot(year=year, show=True)


def print_accounts_info():
    for acc in accounts:
        print(acc.metadata.name.name)
        print(acc.most_recent_date)
        print(acc.get_current_balance())


def plot_predictions(force_new: bool = False):
    start_date = datetime.date(year=2019, month=12, day=2)
    sim_date = datetime.date(year=2022, month=2, day=15)
    end_date = datetime.date(year=2022, month=12, day=31)

    desjardins_mc.plot_prediction_compare(start_date=start_date,
                                          sim_date=sim_date,
                                          end_date=end_date,
                                          show=True, force_new=force_new)


def get_averages():
    avg = accounts.get_daily_average(year=2021)
    print(avg)
    avg = accounts.get_daily_average()
    print(avg)


def accounts_barplots():
    start_date = datetime.date(year=2021, month=12, day=1)
    end_date = datetime.date(year=2021, month=12, day=31)

    accounts.barplot(year=2021, average=False, show=True)
    accounts.barplot(year=2021, average=True, show=True)

    accounts.barplot_date_range(start_date=start_date, end_date=end_date)
    accounts.barplot_date_range(start_date=start_date, end_date=end_date, average=True)

    for i in [10, 11, 12]:
        accounts.barplot(year=2021, month=i, show=True)

    for i in [1]:
        accounts.barplot(year=2022, month=1, show=True)
        accounts.barplot(year=2022, month=1, show=True, average=True)


def bp_years(years, average=False):
    for y in years:
        accounts.barplot(year=y, show=True, average=average)


def bp_last_three_months(average=False):
    d = datetime.datetime.today().date()
    for i in [0, 1, 2]:
        d = d - datetime.timedelta(days=(30 * i))
        accounts.barplot(year=d.year, month=d.month, show=True, average=average)


def bp_current_month():
    accounts.barplot(year=datetime.datetime.today().year,
                     month=datetime.datetime.today().month)


def adhoc_test():
    """ad hoc test"""
    # desjardins_op.update_from_raw_files()


if __name__ == "__main__":
    "tests"
    pred = desjardins_mc.get_predicted_balance(end_date=datetime.date(year=2022, month=4, day=2))
    adhoc_test()
    bp_last_three_months()
    plot_predictions()