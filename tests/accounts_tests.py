import datetime as dt

import matplotlib.pyplot as plt

from accounting import accounts, Account, AccountsList

if __name__ == "__main__":

    fig_name = "All accounts"
    # for account in accounts:
        # print(account.name)
        # account.barplot('2023')
        # account.histplot('2023')
        # account.export()

    accounts.plot(fig_name=fig_name)

    accounts['CIBC'].ignored_index = [690, 692, 703, 708, 714, 717, 727]
    accounts['DesjardinsOP'].ignored_index = [1581, 1586, 1587, 1588, 1673, 1674, 1675, 1676, 1681, 1682]
    accounts['DesjardinsMC'].ignored_index = [454, 455, 487, 488, 491, 492]

    # TODO: get_planned_transaction should planned for payments
    # TODO: predictions with interests
    # TODO: export data and re-import from csv (because of header=0 some first lines are missing),
    #  then lookup for duplicate lines and re-assign code from previous exports

    accounts['NationalBankOP'].legacy_account = accounts
    accounts['NationalBankMC'].legacy_account = accounts['DesjardinsMC']

    accounts['NationalBankOP'].use_legacy_stats = True
    accounts['NationalBankMC'].use_legacy_stats = False

    for sim_date in ['2024-01-08', '2024-01-27', ""]:
        accounts.plot_predictions(predicted_days=int(365 * 1.5),
                                  simulation_date=sim_date,
                                  mc_iterations=100,
                                  figure_name=fig_name,
                                  force_new=False,
                                  show_total=True)
    for year in [
        '2023',
        '2022',
        '2021'
    ]:
        accounts.barplot(year)

    start_date = dt.date(dt.datetime.today().year, dt.datetime.today().month, 1)
    for i in range(3, 0, -1):
        dd = start_date - dt.timedelta(days=1*i)
        start_date = dt.date(dd.year, dd.month, 1)
        accounts.barplot(dd.strftime('%Y-%m'))

    accounts.plot_cumulative_balances(accounts=[accounts['NationalBankOP'],
                                                accounts['NationalBankMC'],
                                                accounts['CIBC']],
                                      fig_name="Cumul. National Bank OP-MC & CIBC")

    plt.show()
