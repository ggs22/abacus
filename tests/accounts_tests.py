import datetime as dt

import matplotlib.pyplot as plt

from accounting import AccountsList, AccountFactory
from accounting.forecast_strategies import (
    PlannedTransactionsStrategy, MonteCarloStrategy, FixedLoanPaymentForecastStrategy,
    CreditCardPaymentForecastStrategy, RequestedForecastList, ForecastFactory
)

account_factory: AccountFactory = AccountFactory()
accounts: AccountsList = account_factory.load_accounts()


if __name__ == "__main__":

    fig_name = "All accounts"
    # for account in accounts:
    #     print(account.name)
    #     account.barplot('2023')
    #     account.histplot('2023')
    #

    accounts.plot(fig_name=fig_name)

    # open_accounts = list()
    # for account in accounts:
    #     if account.status == "OPEN":
    #         open_accounts.append(account)
    # open_accounts = AccountsList(open_accounts)
    # open_accounts.plot(fig_name=fig_name)

    # TODO: implemented outliers-resilient stats
    # TODO: export data and re-import from csv (because of header=0 some first lines are missing),
    #  then lookup for duplicate lines and re-assign code from previous exports

    sim_dates = [""]
    forecasts_factory = ForecastFactory()

    for sim_date in sim_dates:
        if sim_date == "":
            sim_date = dt.date(1900, 1, 1)
            for acc in accounts:
                sim_date = acc.most_recent_date if acc.most_recent_date > sim_date else sim_date
            sim_date = sim_date.strftime("%Y-%m-%d")
        pred_args = {"predicted_days": (dt.date.fromisoformat("2025-12-31")-dt.date.fromisoformat(sim_date)).days,
                     "simulation_date": sim_date,
                     "mc_iterations": 100,
                     "force_new": False,
                     "show_total": True}

        requested_forecasts: RequestedForecastList = [
            (accounts['NationalBankOP'], MonteCarloStrategy(), pred_args),
            (accounts['Paul'], PlannedTransactionsStrategy(), pred_args),
            (accounts['CIBC'], MonteCarloStrategy(), pred_args),
            (accounts['WealthSimpleOP'], PlannedTransactionsStrategy(), pred_args | {"simulation_date": "2025-03-01"}),
            (accounts['WealthSimpleTFSA'], PlannedTransactionsStrategy(), pred_args),
            (accounts['WealthSimpleFHSA'], PlannedTransactionsStrategy(), pred_args),
            (accounts['WealthSimpleCrypto'], PlannedTransactionsStrategy(), pred_args),
            (accounts['IBKRCash'], PlannedTransactionsStrategy(), pred_args),
            (accounts['NationalBankPR'], PlannedTransactionsStrategy(), pred_args)
                     ]

        forecasts = forecasts_factory(requested_forecasts)

        requested_forecasts: RequestedForecastList = [
            (accounts['NationalBankPR'],
             FixedLoanPaymentForecastStrategy(),
             {"op_forecast": forecasts['NationalBankOP'],
              "loan_account": accounts['NationalBankPR'],
              "loan_forecast": forecasts['NationalBankPR'],
              "loan_rate": 0.059,
              "day_of_month": 1,
              "payment_amount": 300}),
            (accounts['NationalBankOP'],
             CreditCardPaymentForecastStrategy(),
             {"op_forecast": forecasts['NationalBankOP'],
              "cc_account": accounts['CIBC'],
              "cc_forecast": forecasts['CIBC']})
        ]

        loan_payment_forecast = forecasts_factory(requested_forecasts, forecasts)

        total_offset = accounts["NationalBankPR"].current_balance
        total_offset += accounts["Paul"].current_balance
        accounts.plot_forecasts(forecasts=loan_payment_forecast, figure_name=fig_name, show_total=True, total_offset=-total_offset)

    start_date = dt.date(dt.datetime.today().year, dt.datetime.today().month, 1)
    for i in range(0, 6):
        dd = start_date - dt.timedelta(days=1*i)
        start_date = dt.date(dd.year, dd.month, 1)
        accounts.barplot(dd.strftime('%Y-%m'))

    accounts.plot_cumulative_balances(accounts=[accounts['NationalBankOP'],
                                                accounts['NationalBankMC'],
                                                accounts['CIBC'],
                                                accounts['WealthSimpleOP'],
                                                accounts['WealthSimpleTFSA'],
                                                accounts['IBKRCash'],
                                                accounts['WealthSimpleFHSA']],
                                      fig_name=fig_name)

    plt.show()
