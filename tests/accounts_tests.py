import matplotlib.pyplot as plt
import pandas as pd

from accounting import accounts

if __name__ == "__main__":

    old_dejardins_op = pd.read_csv(
        r"/home/ggsanchez/repos/abacus/accounting/data/exports/DesjardinsOP/transaction_data_DesjardinsOP.csv",
        sep='\t')
    old_dejardins_op['date'] = pd.to_datetime(old_dejardins_op['date'])
    desjardins_op = accounts['DesjardinsOP'].transaction_data

    name = accounts['DesjardinsOP'].columns_names[(accounts['DesjardinsOP'].columns_names != 'date') &
                                                  (accounts['DesjardinsOP'].columns_names != 'code') &
                                                  (accounts['DesjardinsOP'].columns_names != 'branch') &
                                                  (accounts['DesjardinsOP'].columns_names != 'description')]
    for ix, row in accounts['DesjardinsOP'].filter_by_code(code='na').iterrows():
        idx = (old_dejardins_op[name] == row[name]).all(axis=1)
        row['code'] = old_dejardins_op.loc[idx, 'code'].values[0]

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

    for sim_date in ['']:
        accounts.plot_predictions(predicted_days=182,
                                  simulation_date=sim_date,
                                  mc_iterations=5,
                                  figure_name=fig_name,
                                  force_new=True,
                                  show_total=True)
    for year in [
        '2023',
        '2022',
        '2021'
    ]:
        accounts.barplot(year)

    accounts.barplot("2023-12")

    accounts.plot_cumulative_balances(accounts=[accounts['NationalBankOP'],
                                                accounts['NationalBankMC'],
                                                accounts['CIBC']],
                                      fig_name="Cumul. National Bank OP-MC & CIBC")

    plt.show()
