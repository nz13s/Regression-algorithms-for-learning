from tkinter import Tk, Label, Frame, Button, Entry
from PIL import ImageTk, Image


class GUI:
    def __init__(self):
        self.root = Tk()
        self.root['bg'] = 'orange'
        self.root.title("ZFAC016 - Regression Algorithms for Machine Learning")
        self.root.geometry('700x700')
        self.root.resizable(width=False, height=False)

        author = Label(self.root, text="@author Nick Bogachev", bg='white', font=40)
        author.pack()

        # Set padding
        col_count, row_count = self.root.grid_size()
        for col in range(col_count):
            self.root.grid_columnconfigure(col, minsize=20)
        for row in range(row_count):
            self.root.grid_rowconfigure(row, minsize=20)

        """Datasets buttons"""
        self.data_button_frame = Frame(self.root, bg='orange')
        self.data_button_frame.pack()

        self.irisB = Button(self.data_button_frame, text='Iris')
        self.irisB.grid(row=0, column=0, padx=5, pady=5)

        self.wineB = Button(self.data_button_frame, text='Wine')
        self.wineB.grid(row=0, column=1, padx=5, pady=5)

        self.bostonB = Button(self.data_button_frame, text='Boston')
        self.bostonB.grid(row=0, column=2, padx=5, pady=5)

        self.diabetesB = Button(self.data_button_frame, text='Diabetes')
        self.diabetesB.grid(row=0, column=3, padx=5, pady=5)

        self.importB = Button(self.data_button_frame, text='Import own data')
        self.importB.grid(row=0, column=4, padx=5, pady=5)

        """Model buttons"""
        self.model_button_frame = Frame(self.root, bg='orange')
        self.model_button_frame.pack()

        self.knnB = Button(self.model_button_frame, text='KNN')
        self.knnB.grid(row=0, column=0, padx=5, pady=5)

        self.lsB = Button(self.model_button_frame, text='LS')
        self.lsB.grid(row=0, column=1, padx=5, pady=5)

        self.ridgeB = Button(self.model_button_frame, text='Ridge')
        self.ridgeB.grid(row=0, column=2, padx=5, pady=5)

        self.lassoB = Button(self.model_button_frame, text='Lasso')
        self.ridgeB.grid(row=0, column=3, padx=5, pady=5)

        """Parameter fields"""
        self.param_input_frame = Frame(self.root, bg='orange')
        self.param_input_frame.pack()

        Label(self.param_input_frame, text='K Neighbors').grid(row=0, column=0, padx=5, pady=5)
        self.k_entry = Entry(self.param_input_frame, bg='white')
        self.k_entry.grid(row=0, column=1, padx=5, pady=5)

        Label(self.param_input_frame, text='Alpha for Ridge/Lasso').grid(row=1, column=0, padx=5, pady=5)
        self.a_entry = Entry(self.param_input_frame, bg='white')
        self.a_entry.grid(row=1, column=1, padx=5, pady=5)

        """Graphing buttons"""
        self.graph_button_frame = Frame(self.root, bg='orange')
        self.graph_button_frame.pack()

        self.singleB = Button(self.graph_button_frame, text='Single 2D plot')
        self.singleB.grid(row=0, column=0, padx=5, pady=5)

        self.multiB = Button(self.graph_button_frame, text='Multi-feature plot')
        self.multiB.grid(row=0, column=1, padx=5, pady=5)

        self.knnPlotB = Button(self.graph_button_frame, text='KNN accuracy plot')
        self.knnPlotB.grid(row=0, column=2, padx=5, pady=5)

        self.ridgePlotB = Button(self.graph_button_frame, text='Ridge accuracy plot')
        self.ridgePlotB.grid(row=0, column=3, padx=5, pady=5)

        self.lassoPlotB = Button(self.graph_button_frame, text='Lasso accuracy plot')
        self.lassoPlotB.grid(row=0, column=4, padx=5, pady=5)

        # Plot placement test
        img = Image.open("test_image.png").resize((500, 500), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(self.root, image=img)
        panel.image = img
        panel.pack()


class Main:
    gui = GUI()
    gui.root.mainloop()
