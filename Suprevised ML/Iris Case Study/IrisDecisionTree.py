###################################################################################################
# Required Libraries
###################################################################################################

import os
import sys
import datetime 
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

###################################################################################################
# File path
###################################################################################################
line = "-" * 42
datapath = "iris.csv"
model_path = os.path.join("Artifacts/Iris_DecisionTree", "Iris_DecisionTree_model_v1.joblib")
Folder_path = "Artifacts/Iris_DecisionTree"
file_path = Folder_path + "/" "Iris_DecisionTree_report.txt"

###################################################################################################
# Function name = open_folder_file
# description = This function opens the folder and file for writing.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def Open_Folder_file(File_path = file_path) :
    os.makedirs(Folder_path, exist_ok = True)
    if os.path.exists(File_path) :
        file = open(File_path, 'a')
        
    else :
        
        file = open(File_path, 'w')
    return file

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def read_csv(datapath) :
    return pd.read_csv(datapath)

###################################################################################################
# Function name = Displayhead
# description = This function displays the first few rows of the DataFrame.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def Displayhead(df, file, label="dataset sample is :"):
    file.write(f"{label}\n")
    file.write(df.head().to_string() + "\n\n")

###################################################################################################
# Function name = describe
# description = This function print the description of the dataset.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def describe(datapath, file) :
    file.write(line)
    file.write(datapath.describe().to_string())
    file.write("\n")
    
###################################################################################################
# Function name = columns
# description = This function prints the columns of the dataset.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def columns(datapath, file) :
    file.write(line)
    file.write(datapath.columns.to_series().to_string())
    file.write("\n")

###################################################################################################
# Function name = datatypes
# description = This function prints the datatypes of the dataset.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def datatypes(datapath, file) :
    file.write(line)
    file.write(datapath.dtypes.to_string())
    file.write("\n")

###################################################################################################
# Function name = encoding
# description = This function prints the datatypes of the dataset.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def encoding(datapath) :
        
    for col in datapath.select_dtypes(include='object') :
        datapath[col] = LabelEncoder().fit_transform(datapath[col])
        
    return datapath
    
###################################################################################################
# Function name = DisplayCorrelation
# description = This function displays the correlation matrix of the DataFrame.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def DisplayCorrelation(df, file) :
    file.write("co-relation matrice : \n")
    file.write(df.corr().to_string())
    file.write("\n")

    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
    plt.title("Wine Predictor")
    plt.savefig("Artifacts/Iris_DecisionTree/Correlation_plot.png")
    plt.close()

###################################################################################################
# Function name = DisplayPairplot
# description = This function displays the pairplot of the DataFrame.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def DisplayPairplot(df) :
    sns.pairplot(df)
    plt.suptitle("pairplot of feature", y  = 1.02)
    plt.savefig("Artifacts/Iris_DecisionTree/Pairplot_plot.png")
    plt.close()

###################################################################################################
# Function name = Alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def alter(df) :
    x = df.drop(columns = ['variety'])
    y = df['variety']
    return x, y

###################################################################################################
# Function name = train_model
# description = This function trains the KNN model.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def train_model(x, y) :
    a, b, c, d = train_test_split(x, y, test_size = 0.3)
    return a, b, c, d

###################################################################################################
# Function name = save_model    
# description = This function saves the trained model.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def save_model(model, model_path = model_path) :
    joblib.dump(model, model_path)
    
###################################################################################################
# Function name = load_model
# description = This function loads the trained model.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def load_model(model_path = model_path) :  
    model = joblib.load(model_path)
    return model

###################################################################################################
# Function name = fit_model
# description = This function fits the KNN model.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def fit_model(x_train, y_train) :
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model

###################################################################################################
# Function name = Alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def accuracy_test(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

###################################################################################################
# Function name = DisplayHeatMap
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def DisplayHeatMap(x) :
        
    plt.figure(figsize=(10,6))
    
    sns.heatmap(x.corr(), annot = True, cmap = 'YlGnBu')
    
    plt.title("co-relation between columns")

    plt.savefig("Artifacts/Iris_DecisionTree/HeatMap.png")
    
    plt.close()

    
def Featurn_Importance(model, df) :
    plt.figure(figsize=(12,8))
    plot_tree(model, filled = True, feature_names= df.feature_names, class_names = df.target_names)
    plt.title("Marvllous Decision tree clssification")
    plt.savefig("Artifacts/Iris_DecisionTree/featuer_importance.png")
    
    plt.close()

###################################################################################################
# Function name = Main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################

def main() :
    
    try :
        file = Open_Folder_file()
        
        file.write("date and time for genetaring report is :")
        file.write(datetime.datetime.now())
        
        df = read_csv(datapath)
        
        if sys.argv[1] == "--train" :
            
            Displayhead(df, file)
            
            describe(df, file)
            
            columns(df, file)
            
            datatypes(df, file)
            
            df = encoding(df)
            
            x, y = alter(df)
            
            x_train, x_test, y_train, y_test = train_model(x, y)
            
            model = fit_model(x_train, y_train)
            
            DisplayHeatMap(x)
            
            DisplayPairplot(df)
            
            DisplayCorrelation(df, file)

            save_model(model)
            
            file.write("model is trained and saved \n")
            
        elif sys.argv[1] == "--predict" :
            
            x, y = alter(df)

            x_train, x_test, y_train, y_test = train_model(x, y)
            
            model = fit_model(x_train, y_train)

            model = load_model()

            file.write("model is loaded \n")
    
            y_pred = model.predict(x_test)
                    
            best_accuracy = accuracy_test(y_pred, y_test)
            
            file.write("best accuracy is : ")
            file.write(str(best_accuracy))

        else :
            print("Please provide valid argument : --train or --predict")
            return
            
    except FileNotFoundError as e :
        print(e)
        print("Please provide valid argument : --train or --predict")

###################################################################################################
# application starter
###################################################################################################

if __name__ == "__main__" :
    main()