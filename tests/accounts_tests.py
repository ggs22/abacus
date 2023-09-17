import matplotlib.pyplot as plt

from accounting import accounts


if __name__ == "__main__":
    "tests"

    fig_name = "All accounts"
    for account in accounts:
        print(account.name)
        account.barplot('2023')
        account.histplot('2023')


    accounts.plot(fig_name=fig_name)

    accounts[0].ignored_index = [690, 692, 703, 708, 714, 717, 727]
    accounts[4].ignored_index = [454, 455, 487, 488, 492]

    # for sim_date in ["", "2023-08-31", "2023-06-15"]:
    for sim_date in [""]:
        accounts.plot_predictions(predicted_days=365,
                                  simulation_date=sim_date,
                                  mc_iterations=100,
                                  figure_name=fig_name)

    # accounts.barplot("2023-9")
    accounts.barplot('2023')
    accounts.barplot('2022')
    accounts.barplot('2021')
    accounts.barplot('2020')

    plt.show()
