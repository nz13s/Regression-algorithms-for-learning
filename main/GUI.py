from tkinter import Tk, Label, Frame, Entry, ttk, filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import load_iris, load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from main.KNN import KNN
from main.RegressionModel import RegressionModel


class GUI:
    def __init__(self):
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.k_nbs = 0
        self.alpha = 0

        self.root = Tk()
        self.root['bg'] = 'orange'
        self.root.title("ZFAC016 - Regression Algorithms for Machine Learning")
        self.root.geometry('800x800')
        self.root.resizable(width=True, height=True)

        style = ttk.Style()
        style.map("C.TButton",
                  foreground=[('pressed', 'red'), ('active', 'blue')],
                  background=[('pressed', '!disabled', 'black'),
                              ('active', 'white')])

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

        self.irisB = ttk.Button(self.data_button_frame, text='Iris', command=self.iris_click)
        self.irisB.grid(row=0, column=0, padx=5, pady=5)

        self.wineB = ttk.Button(self.data_button_frame, text='Wine', command=self.wine_click)
        self.wineB.grid(row=0, column=1, padx=5, pady=5)

        self.bostonB = ttk.Button(self.data_button_frame, text='Boston', command=self.boston_click)
        self.bostonB.grid(row=0, column=2, padx=5, pady=5)

        self.diabetesB = ttk.Button(self.data_button_frame, text='Diabetes', command=self.diab_click)
        self.diabetesB.grid(row=0, column=3, padx=5, pady=5)

        """Parameter fields"""
        self.param_input_frame = Frame(self.root, bg='orange')
        self.param_input_frame.pack()

        Label(self.param_input_frame, text='K Neighbors').grid(row=0, column=0, padx=5, pady=5)
        self.k_entry = Entry(self.param_input_frame, bg='white')
        self.k_entry.grid(row=0, column=1, padx=5, pady=5)
        self.k_entry_send = ttk.Button(self.param_input_frame, text='Enter', command=self.grab_k)
        self.k_entry_send.grid(row=0, column=2, padx=5, pady=5)

        Label(self.param_input_frame, text='Alpha for Ridge/Lasso').grid(row=1, column=0, padx=5, pady=5)
        self.a_entry = Entry(self.param_input_frame, bg='white')
        self.a_entry.grid(row=1, column=1, padx=5, pady=5)
        self.a_entry_send = ttk.Button(self.param_input_frame, text='Enter', command=self.grab_a)
        self.a_entry_send.grid(row=1, column=2, padx=5, pady=5)

        """Best params buttons"""
        self.best_frame = Frame(self.root, bg='orange')
        self.best_frame.pack()
        self.k_best = ttk.Button(self.best_frame, text='Get best K for the dataset',
                                 command=self.get_best_k)
        self.k_best.grid(row=0, column=0, padx=5, pady=5)

        self.a_best_ridge = ttk.Button(self.best_frame, text='Get best Ridge alpha for the dataset',
                                       command=self.get_best_ridge)
        self.a_best_ridge.grid(row=0, column=1, padx=5, pady=5)

        self.a_best_lasso = ttk.Button(self.best_frame, text='Get best Lasso alpha for the dataset',
                                       command=self.get_best_lasso)
        self.a_best_lasso.grid(row=0, column=2, padx=5, pady=5)

        """Model buttons"""
        self.model_button_frame = Frame(self.root, bg='orange')
        self.model_button_frame.pack()

        self.knnB = ttk.Button(self.model_button_frame, text='KNN', command=self.knn_click)
        self.knnB.grid(row=0, column=0, padx=5, pady=5)

        self.lsB = ttk.Button(self.model_button_frame, text='Least Squares', command=self.ls_click)
        self.lsB.grid(row=0, column=1, padx=5, pady=5)

        self.ridgeB = ttk.Button(self.model_button_frame, text='Ridge', command=self.ridge_click)
        self.ridgeB.grid(row=0, column=2, padx=5, pady=5)

        self.lassoB = ttk.Button(self.model_button_frame, text='Lasso', command=self.lasso_click)
        self.lassoB.grid(row=0, column=3, padx=5, pady=5)

        """Graphing buttons"""
        self.graph_button_frame = Frame(self.root, bg='orange')
        self.graph_button_frame.pack()

        self.singleB = ttk.Button(self.graph_button_frame, text='Single 2D plot', command=self.single_plot)
        self.singleB.grid(row=0, column=0, padx=5, pady=5)

        self.multiB = ttk.Button(self.graph_button_frame, text='Multi-feature plot', command=self.multi_plot)
        self.multiB.grid(row=0, column=1, padx=5, pady=5)

        self.knnPlotB = ttk.Button(self.graph_button_frame, text='KNN accuracy plot', command=self.knn_accuracy)
        self.knnPlotB.grid(row=0, column=2, padx=5, pady=5)

        self.ridgePlotB = ttk.Button(self.graph_button_frame, text='Ridge accuracy plot', command=self.ridge_accuracy)
        self.ridgePlotB.grid(row=0, column=3, padx=5, pady=5)

        self.lassoPlotB = ttk.Button(self.graph_button_frame, text='Lasso accuracy plot', command=self.lasso_accuracy)
        self.lassoPlotB.grid(row=0, column=4, padx=5, pady=5)

        """Current status bar"""
        self.status_frame = Frame(self.root, bg='orange')
        self.status_frame.pack(pady=5)

        self.status_data = Label(self.status_frame, text='[DATA]')
        self.status_data.grid(row=0, column=0, padx=5, pady=5)

        self.status_model = Label(self.status_frame, text='[MODEL]')
        self.status_model.grid(row=0, column=1, padx=5, pady=5)

        self.status_k = Label(self.status_frame, text='K=...')
        self.status_k.grid(row=0, column=2, padx=5, pady=5)

        self.status_a = Label(self.status_frame, text='Alpha=...')
        self.status_a.grid(row=0, column=3, padx=5, pady=5)

        self.status_score = Label(self.status_frame, text='[SCORE]')
        self.status_score.grid(row=0, column=4, padx=5, pady=5)

        """Reset and save plot"""
        self.reset_frame = Frame(self.root, bg='orange')
        self.reset_frame.pack(pady=5)
        self.reset_button = ttk.Button(self.reset_frame, text='Reset options', command=self.reset)
        self.reset_button.grid(row=0, column=0, padx=5, pady=5)
        self.save_button = ttk.Button(self.reset_frame, text='Save plot', command=self.save_plot)
        self.save_button.grid(row=0, column=1, padx=5, pady=5)

        """Plot frame"""
        self.plot_frame = Frame(self.root, bg='orange')
        self.plot_frame.pack(pady=20, padx=20)

    def iris_click(self):
        self.dataset = load_iris()
        self.status_data.config(text='IRIS')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.data,
                                                                                self.dataset.target,
                                                                                random_state=1512)
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def wine_click(self):
        self.dataset = load_wine()
        self.status_data.config(text='WINE')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.data,
                                                                                self.dataset.target,
                                                                                random_state=1512)
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def boston_click(self):
        self.dataset = load_boston()
        self.status_data.config(text='BOSTON')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.data,
                                                                                self.dataset.target,
                                                                                random_state=1512)
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def diab_click(self):
        self.dataset = load_diabetes()
        self.status_data.config(text='DIABETES')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.data,
                                                                                self.dataset.target,
                                                                                random_state=1512)
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def grab_k(self):
        self.k_nbs = int(self.k_entry.get())
        self.status_k.config(text='K={}'.format(self.k_nbs))
        self.k_entry.delete(0, 'end')

    def get_best_k(self):
        k = 1
        best_knn_score = 0
        best_k = 0
        X_train_pr, X_valid, y_train_pr, y_valid = train_test_split(self.X_train,
                                                                    self.y_train,
                                                                    random_state=1513)
        while k <= len(X_train_pr):
            this_k_knn = KNN(X_train_pr, y_train_pr, k=k)
            this_k_knn.predict(X_valid)
            score = this_k_knn.score(y_valid)
            if score > best_knn_score and score != 1.0:
                best_knn_score = score
                best_k = k
            k += 1
        self.k_entry.insert(0, best_k)

    def grab_a(self):
        self.alpha = float(self.a_entry.get())
        self.status_a.config(text='Alpha={}'.format(self.alpha))
        self.a_entry.delete(0, 'end')

    def get_best_ridge(self):
        alpha_values = np.append(np.arange(0.0, 1.0, 0.01), 1.0)
        best_ridge_score = 0
        best_ridge_a = 0
        X_train_pr, X_valid, y_train_pr, y_valid = train_test_split(self.X_train,
                                                                    self.y_train,
                                                                    random_state=1513)
        for a in alpha_values:
            reg_model = RegressionModel(X_train_pr, y_train_pr)
            reg_model.ridge_fit(alpha=a)
            reg_model.predict(X_valid)
            score = reg_model.score(y_valid)
            if score > best_ridge_score and score != 1.0:
                best_ridge_score = score
                best_ridge_a = a
        self.a_entry.insert(0, round(best_ridge_a, 4))

    def get_best_lasso(self):
        alpha_values = np.append(np.arange(0.0, 1.0, 0.01), 1.0)
        best_lasso_score = 0
        best_lasso_a = 0
        X_train_pr, X_valid, y_train_pr, y_valid = train_test_split(self.X_train,
                                                                    self.y_train,
                                                                    random_state=1513)
        for a in alpha_values:
            reg_model = RegressionModel(X_train_pr, y_train_pr)
            reg_model.lasso_fit(alpha=a)
            reg_model.predict(X_valid)
            score = reg_model.score(y_valid)
            if score > best_lasso_score and score != 1.0:
                best_lasso_score = score
                best_lasso_a = a
        self.a_entry.insert(0, round(best_lasso_a, 4))

    def knn_click(self):
        self.model = KNN(self.X_train, self.y_train, k=self.k_nbs)
        self.status_model.config(text='KNN')

    def ls_click(self):
        self.model = RegressionModel(self.X_train, self.y_train)
        self.model.ls_fit()
        self.status_model.config(text='LEAST SQUARES')

    def ridge_click(self):
        self.model = RegressionModel(self.X_train, self.y_train)
        self.model.ridge_fit(alpha=self.alpha)
        self.status_model.config(text='RIDGE')

    def lasso_click(self):
        self.model = RegressionModel(self.X_train, self.y_train)
        self.model.lasso_fit(alpha=self.alpha)
        self.status_model.config(text='LASSO')

    def single_plot(self):
        self.model.predict(self.X_test)
        X = []
        Y1 = []
        Y2 = []
        fig = plt.figure()
        for row, y1, y2 in zip(self.X_test, self.y_test, self.model.y_pred):
            for item in row:
                X.append(item)
                Y1.append(y1)
                Y2.append(y2)
            plt.scatter(X, Y1, color='black', s=5)
        """
        Before we computed the coefficients array to find y_pred.
        Now it is sort of a reverse process - get 2 coefficients (intercept and slope)
        for a line from the predicted data to plot a best fit line.
        """
        coeffs = np.polyfit(X, Y2, 1)
        m = coeffs[0]
        b = coeffs[1]
        plt.plot(X, np.dot(X, m) + b, color='red')
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.get_tk_widget().pack()
        score = round(self.model.score(self.y_test), 4)
        self.status_score.config(text=score)

    def multi_plot(self):
        self.model.predict(self.X_test)
        # Find number of subplots
        cols = self.X_test.shape[1]
        if cols % 2 == 0:  # if even
            """
            e.g. 12 columns = 12 plots
            12 / 2 = 6
            2 x 6 matrix
            """
            subplot_columns = int((cols / 2))
            subplot_rows = int((cols / subplot_columns))
        else:  # if odd
            """
            e.g. 13 columns = 14 plots
            14 / 2 = 7
            2 x 7 matrix
            """
            round_cols = cols + 1
            subplot_columns = int(round_cols / 2)
            subplot_rows = int(round_cols / subplot_columns)

        fig, axarr = plt.subplots(nrows=subplot_rows, ncols=subplot_columns, sharey=True)
        fig.tight_layout()
        axarr = axarr.flatten()
        X_test_T = self.X_test.T
        for i, column in zip(range(len(axarr)), X_test_T):
            ax = axarr[i]
            X = []
            Y1 = []
            Y2 = []
            for item, y1, y2 in zip(column, self.y_test, self.model.y_pred):
                X.append(item)
                Y1.append(y1)
                Y2.append(y2)
            ax.scatter(X, Y1, color='black', s=5)
            """
            Before we computed the coefficients array to find y_pred.
            Now it is sort of a reverse process - get 2 coefficients (intercept and slope)
            for a line from the predicted data to plot a best fit line.
            """
            coeffs = np.polyfit(X, Y2, 1)
            m = coeffs[0]
            b = coeffs[1]
            ax.plot(X, np.dot(X, m) + b, color='red')
            ax.set_title("Feature {}".format(i + 1), fontsize=10)
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.get_tk_widget().pack()
        score = round(self.model.score(self.y_test), 4)
        self.status_score.config(text=score)

    def knn_accuracy(self):
        k = 1
        X = []
        Y = []
        fig = plt.figure()
        while k <= len(self.X_train):
            this_k_knn = KNN(self.X_train, self.y_train, k=k)
            this_k_knn.predict(self.X_test)
            score = this_k_knn.score(self.y_test)
            X.append(int(k))
            Y.append(score)
            plt.scatter(k, score, color='black', s=20)
            k += 1
        plt.plot(X, Y, color='red')
        plt.xlabel("K neighbors")
        plt.ylabel("R2 scores")
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.get_tk_widget().pack()

    def ridge_accuracy(self):
        alpha_values = np.append(np.arange(0.0, 1.0, 0.01), 1.0)
        scores = []
        fig = plt.figure()
        for a in alpha_values:
            reg_model = RegressionModel(self.X_train, self.y_train)
            reg_model.ridge_fit(alpha=a)
            reg_model.predict(self.X_test)
            score = reg_model.score(self.y_test)
            scores.append(score)
            plt.scatter(a, score, label="Alpha value {}".format(a), color='black', s=20)
        plt.plot(alpha_values, scores, color='red')
        plt.xlabel("Alpha values")
        plt.ylabel("R2 scores")
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.get_tk_widget().pack()

    def lasso_accuracy(self):
        alpha_values = np.append(np.arange(0.0, 1.0, 0.01), 1.0)
        scores = []
        fig = plt.figure()
        for a in alpha_values:
            reg_model = RegressionModel(self.X_train, self.y_train)
            reg_model.lasso_fit(alpha=a)
            reg_model.predict(self.X_test)
            score = reg_model.score(self.y_test)
            scores.append(score)
            plt.scatter(a, score, label="Alpha value {}".format(a), color='black', s=20)
        plt.plot(alpha_values, scores, color='red')
        plt.xlabel("Alpha values")
        plt.ylabel("R2 scores")
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.get_tk_widget().pack()

    def reset(self):
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.k_nbs = 0
        self.alpha = 0
        self.plot_choice = None
        self.k_entry.delete(0, 'end')
        self.a_entry.delete(0, 'end')
        self.status_data.config(text='[DATA]')
        self.status_model.config(text='[MODEL]')
        self.status_k.config(text='K=...')
        self.status_a.config(text='Alpha=...')
        self.status_score.config(text='[SCORE]')
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

    @staticmethod
    def save_plot():
        plot = filedialog.asksaveasfilename(initialdir='/',
                                            title='Save file',
                                            defaultextension='.png',
                                            filetypes=(("PNG Image", "*.png"), ("All Files", "*.*")))
        if plot:
            plt.savefig(plot)
