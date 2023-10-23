from accounting import desjardins_op
from View import View


def quit_abacus():
    exit(0)


def start_gui():
    view = View()

    kwargs = {'year': 2021}
    f = desjardins_op.barplot(**kwargs)
    view.inscribe_dataframe(desjardins_op.get_data(**kwargs))
    view.display_figure(fig=f)

    view.start()


if __name__ == '__main__':

    start_gui()