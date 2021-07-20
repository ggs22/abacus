

import matplotlib.pyplot as plt
import pandas as pd
import PIL
import io
import tkinter as tk

from CSVParser import Account
from tkinter import ttk
from Grapher import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

_tree_view_font_size = 9


class View:

    def __init__(self, csv_parser: CSVParser):
        # set root
        self.root = tk.Tk()
        self.root.title('Abacus')
        self.root.geometry('1600x800')

        # set style
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('Treeview', background='#777777', fieldbackground='#777777', foreground='#cccccc',
                        font=(None, _tree_view_font_size))
        style.configure('Button', bg='#777777', fg='#cccccc')
        self.root.configure(background='#777777')

        self.main_tview = None

        # set external objects dependencies
        self.csv_parser = csv_parser
        self.grapher = Grapher(accounts=csv_parser.accounts)
        self.accounts = self.csv_parser.get_accounts()

        # Custom functions
        self.set_button_action()
        self.set_desjardons_mc_treeview

    def start(self):
        self.root.mainloop()

    def set_button_action(self):
        but_desjardins_op = tk.Button(self.root, text="Desjardins OP", command=self.set_desjardins_op_treeview,
                                      fg='#cccccc', bg='#555555')
        but_desjardins_op.grid(row=0, column=0)

        but_desjardins_mc = ttk.Button(self.root, text="Desjardinc MC", command=self.set_desjardons_mc_treeview)
        but_desjardins_mc.grid(row=1, column=0)

    def inscribe_dataframe(self, df: pd.DataFrame, labels=None):

        if self.main_tview is not None:
            self.main_tview.destroy()

        self.main_tview = ttk.Treeview(self.root, height=35)
        self.main_tview.column('#0', width=0, stretch=tk.NO)

        if labels is None:
            _df = df.copy()
        else:
            _df = df.loc[:, labels].copy()

        _df['date'] = _df['date'].dt.strftime('%Y-%m-%d')

        self.main_tview['columns'] = tuple(_df.columns)
        for col in _df.columns:
            self.main_tview.heading(col, text=col)
            self.main_tview.column(col, width=(len(str(_df.loc[:, col].max())) * _tree_view_font_size) + 5)

        for ix, row in _df.iterrows():
            self.main_tview.insert(parent='', index=tk.END, iid=(row['date'], row['transaction_num']),
                                   values=tuple(row))

        self.main_tview.grid(row=0, column=1, rowspan=50)

    def set_desjardins_op_treeview(self):
        accounts = self.csv_parser.get_accounts()
        self.inscribe_dataframe(accounts['desjardins_op'], labels=['date', 'transaction_num', 'description', 'withdrawal',
                                                       'deposit', 'balance', 'code'])

    def set_desjardons_mc_treeview(self):
        desjardins_mc = self.csv_parser.get_accounts()
        self.inscribe_dataframe(desjardins_mc['desjardins_mc'], labels=['date', 'transaction_num', 'description', 'withdrawal',
                                                       'deposit', 'balance', 'code'])

    def _display_figure(self, fig, row=0, column=0):
        canva = FigureCanvasTkAgg(fig, master=self.root)
        canva.draw()
        canva.get_tk_widget().grid(row=row, column=column)

    def display_desjardins_op(self):
        data = self.accounts['desjardins_op'].get_data_by_date(year=2020)
        f = barplot_desjardins(data=data, fig_size=(5, 2))
        self._display_figure(fig=f, row=0, column=3)

    def display_desjardins_op2(self):
        data = self.accounts['desjardins_op'].get_data_by_date(year=2020, month=1)
        f = barplot_desjardins(data=data, fig_size=(5, 2))
        self._display_figure(fig=f, row=1, column=3)

    def display_desjardins_op3(self):
        data = self.accounts['desjardins_op'].get_data_by_date(year=2020, month=1)
        f = barplot_desjardins(data=data, fig_size=(5, 2))
        self._display_figure(fig=f, row=0, column=4)
