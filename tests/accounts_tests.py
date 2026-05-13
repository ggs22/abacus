import datetime as dt


from accounting import AccountsList, AccountFactory
from accounting.forecast_strategies import (
    MeanTransactionsStrategy, MonteCarloStrategy, FixedLoanPaymentForecastStrategy,
    CreditCardPaymentForecastStrategy, RequestedForecastList, ForecastFactory, PlannedTransactionsStrategy,
    ParallelMonteCarloStrategy
)

account_factory: AccountFactory = AccountFactory()
accounts: AccountsList = account_factory.accounts

if __name__ == "__main__":

    FIG_NAME_ALL = "All accounts"
    FIG_NAME_OPEN = "Open accounts"
    FIG_NAME_PERSONAL = "Personal accounts"

    # TODO: implemented outliers-resilient stats

    sim_dates = [""]
    forecasts_factory = ForecastFactory()

    for sim_date in sim_dates:
        if sim_date == "":
            sim_date = dt.date(1900, 1, 1)
            for acc in accounts:
                sim_date = acc.most_recent_date if acc.most_recent_date > sim_date else sim_date
            sim_date = sim_date.strftime("%Y-%m-%d")
        pred_args = {"predicted_days": (dt.date.fromisoformat("2026-12-31")-dt.date.fromisoformat(sim_date)).days,
                     "simulation_date": sim_date,
                     "mc_iterations": 100,
                     "force_new": True,
                     "show_total": True}

        requested_forecasts: RequestedForecastList = [
            (accounts['NationalBankOP'], PlannedTransactionsStrategy(), pred_args),
            (accounts['Paul'], MeanTransactionsStrategy(), pred_args),
            (accounts['CIBC'], ParallelMonteCarloStrategy(), pred_args),
            (accounts['WealthSimpleOP'], MeanTransactionsStrategy(), pred_args),
            (accounts['WealthSimpleTFSA'], MeanTransactionsStrategy(), pred_args),
            (accounts['WealthSimpleFHSA'], MeanTransactionsStrategy(), pred_args),
            (accounts['WealthSimpleRRSP'], MeanTransactionsStrategy(), pred_args),
            (accounts['WealthSimpleCrypto'], MeanTransactionsStrategy(), pred_args),
            (accounts['IBKRCash'], MeanTransactionsStrategy(), pred_args),
            (accounts['NationalBankPR'], PlannedTransactionsStrategy(), pred_args)
                     ]

        forecasts = forecasts_factory(requested_forecasts)

        requested_forecasts: RequestedForecastList = [
            (accounts['NationalBankPR'],
             FixedLoanPaymentForecastStrategy(),
             {"op_forecast": forecasts['NationalBankOP'],
              "loan_account": accounts['NationalBankPR'],
              "loan_forecast": forecasts['NationalBankPR'],
              "loan_rate": 0.0545,
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

