import datetime as dt

import matplotlib.pyplot as plt

from accounting import AccountsList, AccountFactory
from accounting.forecast_strategies import (
    MeanTransactionsStrategy, MonteCarloStrategy, FixedLoanPaymentForecastStrategy,
    CreditCardPaymentForecastStrategy, RequestedForecastList, ForecastFactory, PlannedTransactionsStrategy,
    ParallelMonteCarloStrategy
)

account_factory: AccountFactory = AccountFactory()
all_accounts: AccountsList = account_factory.accounts

personal_accounts: AccountsList = account_factory.filter_accounts(lambda acc: 'ordial' not in acc.name.lower())


if __name__ == "__main__":

    FIG_NAME_ALL = "All accounts"
    FIG_NAME_OPEN = "Open accounts"
    FIG_NAME_PERSONAL = "Personal accounts"

    all_accounts.plot(fig_name=FIG_NAME_ALL)

    open_accounts = list()
    for account in personal_accounts:
        if account.status == "OPEN":
            open_accounts.append(account)
    open_accounts = AccountsList(open_accounts)
    open_accounts.plot(fig_name=FIG_NAME_OPEN)

    personal_accounts.plot(fig_name=FIG_NAME_PERSONAL)

    # TODO: implemented outliers-resilient stats

    sim_dates = [""]
    forecasts_factory = ForecastFactory()

    for sim_date in sim_dates:
        if sim_date == "":
            sim_date = dt.date(1900, 1, 1)
            for acc in all_accounts:
                sim_date = acc.most_recent_date if acc.most_recent_date > sim_date else sim_date
            sim_date = sim_date.strftime("%Y-%m-%d")
        pred_args = {"predicted_days": (dt.date.fromisoformat("2026-12-31")-dt.date.fromisoformat(sim_date)).days,
                     "simulation_date": sim_date,
                     "mc_iterations": 100,
                     "force_new": True,
                     "show_total": True}

        requested_forecasts: RequestedForecastList = [
            (all_accounts['NationalBankOP'], PlannedTransactionsStrategy(), pred_args),
            (all_accounts['Paul'], MeanTransactionsStrategy(), pred_args),
            (all_accounts['CIBC'], ParallelMonteCarloStrategy(), pred_args),
            (all_accounts['WealthSimpleOP'], MeanTransactionsStrategy(), pred_args),
            (all_accounts['WealthSimpleTFSA'], MeanTransactionsStrategy(), pred_args),
            (all_accounts['WealthSimpleFHSA'], MeanTransactionsStrategy(), pred_args),
            (all_accounts['WealthSimpleRRSP'], MeanTransactionsStrategy(), pred_args),
            (all_accounts['WealthSimpleCrypto'], MeanTransactionsStrategy(), pred_args),
            (all_accounts['IBKRCash'], MeanTransactionsStrategy(), pred_args),
            (all_accounts['NationalBankPR'], PlannedTransactionsStrategy(), pred_args)
                     ]

        forecasts = forecasts_factory(requested_forecasts)

        requested_forecasts: RequestedForecastList = [
            (all_accounts['NationalBankPR'],
             FixedLoanPaymentForecastStrategy(),
             {"op_forecast": forecasts['NationalBankOP'],
              "loan_account": all_accounts['NationalBankPR'],
              "loan_forecast": forecasts['NationalBankPR'],
              "loan_rate": 0.0545,
              "day_of_month": 1,
              "payment_amount": 300}),
            (all_accounts['NationalBankOP'],
             CreditCardPaymentForecastStrategy(),
             {"op_forecast": forecasts['NationalBankOP'],
              "cc_account": all_accounts['CIBC'],
              "cc_forecast": forecasts['CIBC']})
        ]

        loan_payment_forecast = forecasts_factory(requested_forecasts, forecasts)

        total_offset = all_accounts["NationalBankPR"].current_balance
        total_offset += all_accounts["Paul"].current_balance
        all_accounts.plot_forecasts(forecasts=loan_payment_forecast, figure_name=FIG_NAME_OPEN, show_total=True, total_offset=-total_offset)



    personal_accounts.plot_cumulative_balances(accounts=[all_accounts['NationalBankOP'],
                                                         all_accounts['NationalBankMC'],
                                                         all_accounts['CIBC'],
                                                         all_accounts['WealthSimpleOP'],
                                                         all_accounts['WealthSimpleTFSA'],
                                                         all_accounts['WealthSimpleFHSA'],
                                                         all_accounts['WealthSimpleRRSP'],
                                                         all_accounts['IBKRCash']],
                                               fig_name=FIG_NAME_PERSONAL)

    start_date = dt.date(dt.datetime.today().year, dt.datetime.today().month, 1)
    for i in range(0, 7):
        dd = start_date - dt.timedelta(days=1*i*4)
        start_date = dt.date(dd.year, dd.month, 1)
        personal_accounts.barplot(dd.strftime('%Y-%m'))

    plt.show()
