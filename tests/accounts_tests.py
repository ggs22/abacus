import matplotlib.pyplot as plt

from accounting import accounts

if __name__ == "__main__":
    "tests"

    fig_name = "All accounts"
    for account in accounts:
        print(account.name)
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

    for sim_date in ['']:
        accounts.plot_predictions(predicted_days=365*3,
                                  simulation_date=sim_date,
                                  mc_iterations=5,
                                  figure_name=fig_name,
                                  force_new=True,
                                  show_total=False)
    for year in [
        '2023',
        '2022',
        '2021'
    ]:
        accounts.barplot(year)

    plt.show()
