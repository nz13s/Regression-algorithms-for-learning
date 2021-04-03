Date: 03.04.2021, for version: v2.1

The following project runs on Python 3.7, requires NumPy, Matplotlib, and Sklearn libraries as main dependancies (the full list is available in requirements.txt) and consists of two subdirectories - "main" and "test":
- "main" contains the currently implemented algorithms - KNN and Regression Model (includes Least Squares, Ridge and Lasso) - in the form of classes with the expected _init_, fit, predict and score methods. Additionally, "main" contains the GUI created with TKInter.
- "test" contains test classes for each algorithm accordingly.

The file that needs to be executed is "ZFAC016_main_run.py". 

Due to Python not being able to create an executable in the same efficient manner Java does via a .jar file, to be able to use this project it is best to create a virtual environment in the terminal.

If you do not have Python 3 installed, it is best to do so beforehand. The current version can be installed from https://www.python.org/downloads/. 
Check you have Python and PIP successfully installed by opening a Terminal window on your machine and typing the following:
- "python3 -V" to confirm you have Python installed and running a correct version (3.7.x)
- "pip3 -V" to confirm you have PIP installed.

Having Terminal opened, navigate to the directory where you have saved this project to:
<username>:cd path/to/project
e.g. if you saved it to Downloads on MacOS, it would be "cd Downloads/IndividualProject_2020_Nikita-Bogachev/".

Now you can create a virtual environment (or venv):
- For Mac and Linux: 
  python3 -m venv env
  source env/bin/activate
- For Windows: 
  py -m venv env
  .\env\Scripts\activate
  
Now that you have created and activated the virtual environment, install the required modules into it:
- Mac and Linux: python3 -m pip install -r requirements.txt
- Windows: py -m pip install -r requirements.txt

After the requirements have been installed, you can tell Python to execute the script mentioned above:
- Mac and Linux: python3 ZFAC016_main_run.py
- Windows: py ZFAC016_main_run.py

After a short wait, you will see a GUI in front of you. It may seem daunting at first from all the button and fields, but it is a fairly simple design. You will see the following interaction areas:
- Row 1: Four datasets for your choice. Iris and Wine work better with KNN and Boston and Diabetes work better with the other three algorithms.
- Row 2: Entry field for a value of K for K Nearest Neighbors algorithm. The result can be saved to the status bar in Row 7 by clicking Enter.
- Row 3: Entry field for a value of alpha for Ridge and Lasso algorithms. The result can be saved to the status bar in Row 7 by clicking Enter.
- Row 4: Buttons to get the best values for the currently chosen dataset. They won't work if the dataset is not chosen! The result will be printed in the fields of Row 2 and 3 and then must be saved to the status bar of Row 7 by clicking Enter.
- Row 5: Algorithm selection. Choosing one should only happen AFTER you have chosen alpha or K and saved it to the status bar, otherwise the algorithm won't be able to make a prediction matrix.
- Row 6: Choose what plot you wish to see the line of best fit drawn on.
  - single 2D plot will make a scatter plot of all data in 2D, sometimes with points on top of each other, and draw a line of best fit thought it. Must have DATA, MODEL and K/Alpha chosen in the status bar.
  - multi-feature will build a scatter plot for each feature of the matrix. Each column represent a feature, so each graph will show a given column values with the corresponding y-labels scattered wwith a line of best fit going through them. Must have DATA, MODEL and K/Alpha chosen in the status bar.
  - Three accuracy plots to see how the algorithm works on your currently selected dataset with increasing k or alpha. Must have DATA chosen in the status bar.
- Row 7: The status bar. This is to track your interactions and confirm that the system have saved your choices. Depending on actions currently being performed you can compare the above instructions to this status bar to make sure you have given correct arguments to the system.
- Row 8: Reset your current options and save plot button. Reset button must be clicked before you make a new plot as the frame does not update without a full reset. Additionally, if you think you've made a mistake or something does not work this is works as a safe way to start over. The option to save the currently displayed plot will result in you being prompted with a window that defaults the file type to .png and asks you to select the directory where you would like to save the file and name it. After the GUI is closed, the directory will update with saved plots.

An example output and order of interactions:
- Choose Boston dataset. You will see [DATA] change to BOSTON.
- Lets say you type 0.8 in "Alpha for Ridge/Lasso" and click Enter. You will see Alpha=... change to Alpha=0.8
- This may be too high. Choose "Get best Lasso alpha for the dataset". You will see the field you've just used update with the float 0.04. Click Enter and you will see Alpha=0.8 change to Alpha=0.04.
- Now select Lasso and you will see [MODEL] change to LASSO.
- Click Multi-feature plot. The plot will be shown below and the [SCORE] in the status bar will update accordingly.
- Click Reset options to make a new plot.
