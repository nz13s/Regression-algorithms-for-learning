from tkinter import Tk, Label, Frame, Button, Entry


class GUI:
    def __init__(self):
        self.root = Tk()
        self.root['bg'] = 'orange'
        self.root.title("Regression Algorithm Analysis")
        self.root.geometry('650x500')
        self.root.resizable(width=False, height=False)
        self.selected_data = None

        author = Label(self.root, text="@author Nick Bogachev", bg='white', font=40)
        author.pack()

        # Datasets buttons
        self.data_button_frame = Frame(self.root, bg='orange')
        self.data_button_frame.pack()
        Button(self.data_button_frame, text='Iris').grid(row=0, column=0)
        Button(self.data_button_frame, text='Wine').grid(row=0, column=1)
        Button(self.data_button_frame, text='Boston').grid(row=0, column=2)
        Button(self.data_button_frame, text='Diabetes').grid(row=0, column=3)
        Button(self.data_button_frame, text='Import own data').grid(row=0, column=4)

        # Model buttons
        self.model_button_frame = Frame(self.root, bg='orange')
        self.model_button_frame.pack()
        Button(self.model_button_frame, text='KNN').grid(row=0, column=0)
        Button(self.model_button_frame, text='LS').grid(row=0, column=1)
        Button(self.model_button_frame, text='Ridge').grid(row=0, column=2)
        Button(self.model_button_frame, text='Lasso').grid(row=0, column=3)

        # Parameter fields
        self.param_input_frame = Frame(self.root, bg='orange')
        self.param_input_frame.pack()
        Label(self.param_input_frame, text='K Neighbors').grid(row=0, column=0)
        Entry(self.param_input_frame, bg='white').grid(row=0, column=1)
        Label(self.param_input_frame, text='Alpha for Ridge/Lasso').grid(row=1, column=0)
        Entry(self.param_input_frame, bg='white').grid(row=1, column=1)

        # Graphing buttons
        self.graph_button_frame = Frame(self.root, bg='orange')
        self.graph_button_frame.pack()
        Button(self.graph_button_frame, text='Single 2D plot').grid(row=0, column=0)
        Button(self.graph_button_frame, text='Multi-feature plot').grid(row=0, column=1)
        Button(self.graph_button_frame, text='KNN accuracy plot').grid(row=0, column=2)
        Button(self.graph_button_frame, text='Ridge accuracy plot').grid(row=0, column=3)
        Button(self.graph_button_frame, text='Lasso accuracy plot').grid(row=0, column=4)


class Main:
    gui = GUI()
    gui.root.mainloop()
