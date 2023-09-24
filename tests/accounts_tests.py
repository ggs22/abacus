import matplotlib.pyplot as plt

from accounting import accounts
from accounting.prediction_strategies import PredictionByMeanStrategy, BasicMonteCarloStrategy


if __name__ == "__main__":
    "tests"

    fig_name = "All accounts"
    for account in accounts:
        print(account.name)
        # account.barplot('2023')
        # account.histplot('2023')
        # account.export()

    accounts.plot(fig_name=fig_name)

    accounts[0].ignored_index = [690, 692, 703, 708, 714, 717, 727]
    accounts[4].ignored_index = [454, 455, 487, 488, 492]

    # for sim_date in ["", "2023-08-31", "2023-06-15"]:
    for strategy in [BasicMonteCarloStrategy, PredictionByMeanStrategy]:
        for sim_date in [""]:
            accounts.plot_predictions(predict_strategy=strategy(),
                                      predicted_days=365,
                                      simulation_date=sim_date,
                                      mc_iterations=100,
                                      figure_name=fig_name)
    for year in [
        '2023',
        '2022',
        '2021'
    ]:
        accounts.barplot(year)

    plt.show()
