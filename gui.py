from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from pathlib import Path
home = str(Path.home())

root = Tk()

def OpenFile():
    name = askopenfilename(initialdir=home,
                           filetypes =(("Image File", "*.png *.jpg *.jpeg *.bmp *.gif *.tif"),("All Files","*.*")),
                           title = "Choose a file."
                           )
    print (name)
    #Using try in case user types in unknown file or closes without choosing a file.
    try:
        with open(name,'r') as UseFile:
            print(UseFile.read())
    except:
        print("No file exists")

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

Title = root.title( "File Opener")

#Menu Bar

menu = Menu(root)
root.config(menu=menu)

file = Menu(menu)

file.add_command(label = 'Open', command = OpenFile)
file.add_command(label = 'Exit', command = lambda:exit())

menu.add_cascade(label = 'File', menu = file)

root.mainloop()