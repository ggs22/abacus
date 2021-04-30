from tkinter import *
from PIL import Image, ImageTk


class View:

    def __init__(self):
        self.root = Tk()
        self.root.title('Abacus')
        self.root.geometry('500x1200')


    def start(self):
        self.root.mainloop()


#     '''
#         TKinter start
#         '''
#
#     # creating the label widget
#     lbl1 = Label(self.root, text='This and that')
#
#     btn1 = Button(self.root, text='Graph', padx=50, pady=50,
#                   command=grapher.plot_desjardins_mc, fg='#0000ff', bg='black')
#
#     # btn1 = Button(self.root, text='Graph', padx=50, pady=50,
#     #               command=lambda: grapher.plot_desjardins_mc(**kwargs), fg='#0000ff', bg='black')
#
#     # 1 pack (unsophistiated): shoving it onto the screen
#     # lbl.pack()
#
#     # 2 grid
#     row = 0
#     lbl1.grid(row=0, column=0)
#
#     for ix, comm in enumerate(commnands):
#         btn = Button(self.root, text=comm.name, command=comm.method, fg='#0000ff')
#         row += 1
#         btn.grid(row=row, column=0)
#
#     img = Image.open('images/total.png')
#     # img = img.resize((100, 100), Image.ANTIALIAS)
#     img = ImageTk.PhotoImage(img)
#     lbl2 = Label(self.root, image=img)
#     lbl2.grid(row=1, column=1, rowspan=100)
#
#     cashflow = Listbox(self.root, width=50)
#     for ix, order in csv_parser.get_data()[0].iterrows():
#         cashflow.insert(ix, order.values)
#     row += 1
#     cashflow.grid(row=row, column=0)
#
# '''
# TKinter end
# '''
