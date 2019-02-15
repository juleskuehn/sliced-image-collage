from tkinter import *

root = Tk()

# Prefixing t_ for text, b_ for button elements, c_ for checkbox etc
t_title_1 = Label(root, text='kWord')
b_select_target = Button(root, text='Select target image...')
c_dither_on = Checkbutton(root, text='Dither')
e_dither_amt = Entry(root)
l_dither_amt = Label(root, text='Amount')

t_title_1.grid(row=0, columnspan=2)
b_select_target.grid(row=1, columnspan=2)
l_dither_amt.grid(row=2, column=1, sticky=E)
e_dither_amt.grid(row=2, column=2)
c_dither_on.grid(row=2, column=0)
root.mainloop()