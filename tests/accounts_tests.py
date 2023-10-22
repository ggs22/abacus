import matplotlib.pyplot as plt

from accounting import accounts
from accounting.prediction_strategies import (
    PredictionByMeanStrategy, DateBasedMonteCarloStrategy
)


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

    # TODO - account-specific strategies (List[PredictionStrategy])
    # TODO - planned transaction & mean Strategy to implement
    for strategy in [
        PredictionByMeanStrategy,
        DateBasedMonteCarloStrategy,
    ]:
        for sim_date in ['2023-09-30', ""]:
            accounts.plot_predictions(predict_strategy=strategy(),
                                      predicted_days=365*3,
                                      simulation_date=sim_date,
                                      mc_iterations=50,
                                      figure_name=fig_name,
                                      force_new=False)
    for year in [
        '2023',
        '2022',
        '2021'
    ]:
        accounts.barplot(year)

    plt.show()
