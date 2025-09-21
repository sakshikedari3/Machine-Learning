###################################################################################################
# Required Libraries
###################################################################################################
import joblib
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

###################################################################################################
# File path
###################################################################################################
line = "-" * 42
dataset = "Advertising.csv"
model_path = "advertising_model.joblib"
Folder_path = "Artifacts/Advertisement"
file_path = Folder_path + "/" "advertisement_report.txt"

###################################################################################################
# Function name = open_folder_file
# description = This function opens the folder and file for writing.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def Open_Folder_file(file_path) :
    os.makedirs(Folder_path, exist_ok = True)
    file = open(file_path, 'w')
    return file

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def read_csv(datapath) :
    return pd.read_csv(datapath)

###################################################################################################
# Function name = Displayhead
# description = This function displays the first few rows of the DataFrame.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def Displayhead(df, file, label="dataset sample is :"):
    file.write(f"{label}\n")
    file.write(df.head().to_string() + "\n\n")

###################################################################################################
# Function name = describe
# description = This function print the description of the dataset.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def describe(datapath, file) :
    file.write(line)
    file.write(datapath.describe().to_string())
    file.write("\n")

###################################################################################################
# Function name = columns
# description = This function prints the columns of the dataset.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def columns(datapath, file) :
    file.write(line)
    file.write(datapath.columns.to_series().to_string())
    file.write("\n")

###################################################################################################
# Function name = datatypes
# description = This function prints the datatypes of the dataset.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def datatypes(datapath, file) :
    file.write(line)
    file.write(datapath.dtypes.to_string())
    file.write("\n")

###################################################################################################
# Function name = encode_categorical
# description = This function encodes categorical variables in the DataFrame.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def encode_categorical(df) :
    df.drop(columns = ['Unnamed: 0'], inplace = True) #cut that column on that space
    return df

###################################################################################################
# Function name = FindMissingValues
# description = This function finds missing values in the DataFrame.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def FindMissingValues(df, file) :
    file.write("missing values in each columns  :\n")
    file.write(df.isnull().sum().to_string())
    file.write("\n")

###################################################################################################
# Function name = DisplayCorrelation
# description = This function displays the correlation matrix of the DataFrame.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def DisplayCorrelation(df, file) :
    file.write("co-relation matrice : \n")
    file.write(df.corr().to_string())
    file.write("\n")

    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
    plt.title("advertisement")
    plt.savefig("Artifacts/Advertisement/Correlation_plot.png")
    plt.close()
   
###################################################################################################
# Function name = DisplayPairplot
# description = This function displays the pairplot of the DataFrame.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def DisplayPairplot(df) :

    sns.pairplot(df)
    plt.suptitle("pairplot of feature", y  = 1.02)
    plt.savefig("Artifacts/Advertisement/Pairplot_plot.png")
    plt.close()
  
###################################################################################################
# Function name = Alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def Alter(df) :
    x = df[['TV', 'radio','newspaper']]
    y = df['sales']
    return x, y

###################################################################################################
# Function name = train_model
# description = This function trains the model using the training data.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def train_model(x, y) :
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

###################################################################################################
# Function name = model_fit
# description = This function fits the model to the training data.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def model_fit(x_train, y_train) :
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

###################################################################################################
# Function name = Save_model
# description = This function saves the trained model to a file.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def save_model(model, file) :
    joblib.dump(model, model_path)
    file.write(f"model saved at {model_path}\n")

###################################################################################################
# Function name = load_model
# description = This function loads the trained model from a file.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def load_model() :
    model = joblib.load(model_path)
    return model
    
###################################################################################################
# Function name = MeanSquareError
# description = This function calculates the Mean Squared Error.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def MeanSquareError(y_test, y_pred) :
    return mean_squared_error(y_test, y_pred)

###################################################################################################
# Function name = RootMeanSquareError
# description = This function calculates the Root Mean Squared Error.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def RootMeanSquareError(mse) :
    return np.sqrt(mse)

###################################################################################################
# Function name = R2Score
# description = This function calculates the R2 Score.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def R2Score(y_test, y_pred) :
    return r2_score(y_test, y_pred)

###################################################################################################
# Function name = display_coefficients
# description = This function displays the coefficients of the model.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def display_coefficients(model, x, file) :
    file.write('model coefficients are : \n')
    for col, coef in zip(x.columns, model.coef_) :
        file.write(f"{col} : {coef}\n")

    file.write(f"intercept c : {model.intercept_}\n")
    
###################################################################################################
# Function name = display_plot
# description = This function displays a scatter plot of actual vs predicted values.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def DisplayPlot(y_test, y_pred) :
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color = 'blue')
    plt.xlabel("actual sale")
    plt.ylabel("predicted sale")
    plt.title("advertisement")
    plt.grid(True)
    plt.savefig("Artifacts/Advertisement/prediction_plot.png")
    plt.close()
###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def main() :

    file = Open_Folder_file(file_path)

    df = read_csv(dataset)
    
    Displayhead(df, file)
    
    describe(df, file)
    
    columns(df, file)
    
    datatypes(df, file)
    
    df = encode_categorical(df)
    
    file.write("updated data is : ")
    Displayhead(df, file)

    FindMissingValues(df, file)

    DisplayCorrelation(df, file)

    DisplayPairplot(df)

    x, y = Alter(df)

    file.write("features are : \n")
    file.write(str(x))
    file.write("target is : \n")
    file.write(str(y))

    x_train, x_test, y_train, y_test = train_model(x, y)

    model = model_fit(x_train, y_train)
    
    save_model(model, file)
    
    model = load_model()
    
    y_pred = model.predict(x_test)
    
    mse = MeanSquareError(y_test, y_pred)
    file.write("\nmean square error is :")
    file.write(str(mse))

    rmse = RootMeanSquareError(mse)
    file.write("\nroot mean square error is :")
    file.write(str(rmse))

    r2 = R2Score(y_test, y_pred)
    file.write("\nR square value is :")
    file.write(str(r2))

    display_coefficients(model, x, file)
    
    DisplayPlot(y_test, y_pred)

###################################################################################################
# application starter
###################################################################################################

if __name__ == "__main__" :
    main()
