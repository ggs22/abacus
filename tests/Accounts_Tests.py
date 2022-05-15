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
    sim_date = datetime.date(year=2022, month=3, day=17)
    end_date = datetime.date(year=2023, month=11, day=30)

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

    accounts.barplot(year=2021, show=True)

    accounts.barplot_date_range(start_date=start_date, end_date=end_date)

    for i in [10, 11, 12]:
        accounts.barplot(year=2021, month=i, show=True)

    for i in [1]:
        accounts.barplot(year=2022, month=1, show=True)


def bp_years(years):
    for y in years:
        accounts.barplot(year=y, show=True)


def bp_last_months(average=False, num_months: int = 3 ):
    d = datetime.datetime.today().date()
    mi = d.month
    y = d.year
    count = 0
    for i in range(0, num_months):
        m = (mi - count) % 12
        m = m * (m > 0) + 12 * (m == 0)
        if m == 12:
            y = (y - 1) * (m == 12) + y * (m < 12)
        accounts.barplot(year=y, month=m, show=True)
        count += 1


def bp_current_month():
    accounts.barplot(year=datetime.datetime.today().year,
                     month=datetime.datetime.today().month)


if __name__ == "__main__":
    "tests"
    # pred = desjardins_mc.get_predicted_balance(sim_date=datetime.date(year=2022, month=3, day=31),
    #                                            end_date=datetime.date(year=2023, month=12, day=31))
    bp_last_months(num_months=7)
    plot_predictions(force_new=False)
