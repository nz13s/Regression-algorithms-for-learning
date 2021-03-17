from tkinter import *
from tkinter import messagebox

root = Tk()
root['bg'] = '#fafafa'
root.title("Test file")
root.geometry('300x250')
root.resizable(width=False, height=False)

canvas = Canvas(root, height=300, width=250)
canvas.pack()

frame = Frame(root, bg='red')
frame.place(relx=0.15, rely=0.15, relwidth=0.7, relheight=0.7)

title = Label(frame, text='Hint', bg='grey', font=40)
title.pack()


def btn_click():
    login = loginInput.get()
    password = passField.get()
    info_str = f'Data: {str(login)}, {str(password)}'
    messagebox.showinfo(title='Name', message=info_str)


btn = Button(frame, text='Button', bg='yellow', command=btn_click)
btn.pack()

loginInput = Entry(frame, bg='white')
loginInput.pack()
passField = Entry(frame, bg='white', show="*")
passField.pack()

root.mainloop()
