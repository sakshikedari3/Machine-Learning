###################################################################################################
# Required Libraries
###################################################################################################s
import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

###################################################################################################
# File Path and Model Path
###################################################################################################s

line = "*" * 44
datapath = "HeadBrain.csv"
model_path = "HeadBrainLinearModel.joblib"
folder_path = "Artifacts/HeadBrain"
file_path = folder_path + "/HeadBrain.txt"

###################################################################################################
# Function name = Open_File
# description = This function opens a file and returns the file object.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################

def Open_File():
    os.makedirs(folder_path, exist_ok=True)
    file = open(file_path, "w")
    return file

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def ReadCSV(DATAPATH):
    return pd.read_csv(DATAPATH)

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def DisplayHead(df, file):
    file.write("first few records of the data set are : \n")
    file.write(line + "\n")
    file.write(df.head().to_string() + "\n")
    file.write(line + "\n")

###################################################################################################
# Function name = DisplayStatisticalInfo
# description = This function displays the statistical information of the DataFrame.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def DisplayStatisticalInfo(df, file):
    file.write("statistical information are : \n")
    file.write(line + "\n")
    file.write(df.describe().to_string() + "\n")
    file.write(line + "\n")

###################################################################################################
# Function name = HeatMap
# description = This function displays the heatmap of the correlation matrix of the DataFrame.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
    
def HeatMap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Heatmap of HeadBrain Dataset")
    plt.savefig("Artifacts/HeadBrain/Heatmap.png")
    plt.close()
    
###################################################################################################
# Function name = alter
# description = This function alters the DataFrame.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def alter(df):
    x = df[['Head Size(cm^3)']]
    y = df[['Brain Weight(grams)']]
    return x, y

###################################################################################################
# Function name = scaler
# description = This function scales the features using StandardScaler.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def scaler(x) :
    return StandardScaler().fit_transform(x)

###################################################################################################
# Function name = printSize
# description = This function prints the size of the independent and dependent variables.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def PrintSize(x, y, file):
    file.write("independent variables are : Head Size\n")
    file.write("dependent variables are : Brain Weight\n")
    file.write("total records in data set : " + str(x.shape) + "\n")
    file.write("total records in data set : " + str(len(x)) + "\n")
    file.write("total records in data set : " + str(y.shape) + "\n")
    file.write("total records in data set : " + str(len(y)) + "\n")
    
###################################################################################################
# Function name = printSize
# description = This function prints the size of the independent and dependent variables.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def TrainTestSplitDf(x, y):
    a, b, c, d = train_test_split(x, y, test_size = 0.2, random_state = 42)
    return a, b, c, d

###################################################################################################
# Function name = PrintTrainingTestSize
# description = This function prints the size of the training and testing datasets.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def PrintTrainingTestSize(x_train, x_test, y_train, y_test, file):
    file.write("dimentions of traing data set : " + str(len(x_train)) + "\n")
    file.write("dimentions of traing data set : " + str(len(x_test)) + "\n")
    file.write("dimentions of traing data set : " + str(len(y_train)) + "\n")
    file.write("dimentions of traing data set : " + str(len(y_test)) + "\n")

###################################################################################################
# Function name = fit
# description = This function fits the linear regression model.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def fit(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model
###################################################################################################
# Function name = save_model
# description = This function saves the trained model using joblib.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
###################################################################################################
# Function name = load_model
# description = This function loads the trained model using joblib.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def load_model(model_path):
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)

###################################################################################################
# Function name = mean_square_error
# description = This function calculates the mean square error.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def mean_square_error(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)

###################################################################################################
# Function name = root_mean_square_error
# description = This function calculates the root mean square error.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def root_mean_square_error(mse):
    return np.sqrt(mse)

###################################################################################################
# Function name = r2score
# description = This function calculates the R^2 score.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################
def r2score(y_test, y_pred):
    return r2_score(y_test, y_pred)

###################################################################################################
# Function name = ScatterPlot
# description = This function creates a scatter plot of the actual vs predicted values.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################

def scatterPlot(x_test, y_test, y_pred):

    print("visual represention")

    plt.figure(figsize=(8,5))

    plt.scatter(x_test, y_test, color = 'blue', label = 'actual')

    plt.plot(x_test, y_pred, color = 'red', linewidth = 2, label = 'regression line')

    plt.xlabel("Head Size(cm^3)")

    plt.ylabel("Brain Weight(grams)")

    plt.title("head-brain ")

    plt.legend()

    plt.grid(True)

    plt.savefig("Artifacts/HeadBrain/ScatterPlot.png")
    
    plt.close()
    
###################################################################################################
# Function name = Main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 01-10-2025
###################################################################################################

def main() :
    
    file = Open_File()
    #1) read csv file
    df = ReadCSV(datapath)
    
    x, y = alter(df)
    
    x = scaler(x)
    
    try :
        if sys.argv[1] == "--train" :
    
            #2) display first few records
            DisplayHead(df, file)
            
            #3) display statistical information
            DisplayStatisticalInfo(df, file)

            #4) heat map
            HeatMap(df)
            
            #7) print size of independent and dependent variables
            PrintSize(x, y, file)

            #8) split the data into training and testing datasets
            x_train, x_test, y_train, y_test = TrainTestSplitDf(x, y)
            
            #9) print size of training and testing datasets
            PrintTrainingTestSize(x_train, x_test, y_train, y_test, file)

            #10) fit the model
            model = fit(x_train, y_train)

            #11) predict the model
            prediction = model.predict(x_test)

            #12) save the model
            save_model(model, model_path)
            
        elif sys.argv[1] == "--predict" :
            
            x_train, x_test, y_train, y_test = TrainTestSplitDf(x, y)
            
            #13) load the model
            model1 = load_model(model_path)

            #14) predict using the loaded model
            prediction = model1.predict(x_test)

            #15) calculate and print the evaluation metrics
            mse = mean_square_error(y_test, prediction)
            file.write("Mean Square Error:")
            file.write(str(mse) + "\n")

            #16) calculate and print the root mean square error
            rmse = root_mean_square_error(mse)
            file.write("Root Mean Square Error: " )
            file.write(str(rmse) + "\n")            
            
            #17) calculate and print the R^2 score
            r2 = r2score(y_test, prediction)
            file.write("R^2 Score: ")
            file.write(str(r2) + "\n")

            #18) visualize the results using scatter plot
            scatterPlot(x_test, y_test, prediction)
            
        else:
            print("Invalid argument. Use --train or --predict.")
            exit(1)
            
    except Exception as e :
        print(e)
        exit(1)
    
###################################################################################################
# application starter
###################################################################################################
if __name__ == "__main__" :
    main()