###################################################################################################
# Required Libraries
###################################################################################################s

import os
import sys
import joblib
import pandas as pd
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

###################################################################################################
# File Path and Model Path
###################################################################################################s

line = "*" * 44
datapath = "weight-height.csv"
model_path = "Artifacts/Weight Height/Weight_HeightLogisticModel.joblib"
folder_path = "Artifacts/Weight Height"
file_path = folder_path + "/Weight_HeightLogistic.txt"

os.makedirs(folder_path, exist_ok=True)
file = open(file_path, "w")

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def ReadCSV(DATAPATH):
    return pd.read_csv(DATAPATH)

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def DisplayShape(df, file = file) :
    file.write(f"dimention of dataset is : {df.shape}\n")
    file.write(line + "\n")

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def DisplayHead(df, file = file):
    file.write("first few records of the data set are : \n")
    file.write(line + "\n")
    file.write(df.head().to_string() + "\n")
    file.write(line + "\n")

###################################################################################################
# Function name = DisplayStatisticalInfo
# description = This function displays the statistical information of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def DisplayStatisticalInfo(df, file = file):
    file.write("statistical information are : \n")
    file.write(line + "\n")
    file.write(df.describe().to_string() + "\n")
    file.write(line + "\n")
    
###################################################################################################
# Function name = Mapping
# description = This function map and encode manually
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################
    
def Mapping(df) :
    df['Gender'] = df['Gender'].map({'Female' : 0, 'Male' : 1})
    return df

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def DisplayCountPlot(df):
    plt.figure()
    target = "Gender"
    sns.countplot(data = df, x = target).set_title("survived vs non survive")
    plt.savefig("Artifacts/TitanicLogisticRegression/survived_vs_non_survive.png")
    plt.close()

    plt.figure()
    target = "Gender"
    sns.countplot(data = df, x = target, hue = 'Height').set_title("based on height")
    plt.savefig("Artifacts/TitanicLogisticRegression/survived_vs_gender.png")
    plt.close()

    plt.figure()
    target = "Gender"
    sns.countplot(data = df, x = target, hue = 'Weight').set_title("based on weight")
    plt.savefig("Artifacts/TitanicLogisticRegression/survived_vs_pclass.png")
    plt.close()

    plt.figure()
    df['Height'].plot.hist().set_title("age report")
    plt.savefig("Artifacts/TitanicLogisticRegression/age_report.png")
    plt.close()

    plt.figure()
    df['Weight'].plot.hist().set_title("fare report")
    plt.savefig("Artifacts/TitanicLogisticRegression/fare_report.png")
    plt.close()

###################################################################################################
# Function name = HeatMap
# description = This function displays the heatmap of the correlation matrix of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################
    
def HeatMap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Heatmap of HeadBrain Dataset")
    plt.savefig("Artifacts/Weight Height/Weight_HeightLogistic_HeatMap.png")
    plt.close()
    
###################################################################################################
# Function name = ScatterPlot
# description = This function displays the scatter plot of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################
    
def ScatterPlot(df):
    plt.figure(figsize = (8,6))
    sns.scatterplot(data = df, x = 'Height', y = 'Weight', hue = 'Gender', palette = 'Set1')
    plt.title("Marvellous weight_height")
    plt.xlabel("height")
    plt.ylabel("weight")
    plt.savefig("Artifacts/Weight Height/Weight_HeightLogistic_ScatterPlot.png")
    
###################################################################################################
# Function name = alter
# description = This function alters the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################
def alter(df):
    x = df[['Height', 'Weight']]
    y = df['Gender']
    return x, y

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def display_dimention(x, y, file = file):
    file.write("dimentation of feature : ")
    file.write(str(x.shape))
    file.write("dimentation of label : ")
    file.write(str(y.shape))
    file.write(line + "\n")

###################################################################################################
# Function name = scaler
# description = This function scales the features using StandardScaler.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def scaler(x) :
    return StandardScaler().fit_transform(x)

###################################################################################################
# Function name = printSize
# description = This function prints the size of the independent and dependent variables.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def PrintSize(x, y, file = file):
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
# date = 07-10-2025
###################################################################################################

def TrainTestSplit(x, y):
    a, b, c, d = train_test_split(x, y, test_size = 0.2, random_state = 42)
    return a, b, c, d

###################################################################################################
# Function name = PrintTrainingTestSize
# description = This function prints the size of the training and testing datasets.
# author = sakshi kedari
# date = 07-10-2025
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
# date = 07-10-2025
###################################################################################################

def fit(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model
###################################################################################################
# Function name = save_model
# description = This function saves the trained model using joblib.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
###################################################################################################
# Function name = load_model
# description = This function loads the trained model using joblib.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def load_model(model_path):
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def Accuracy(y_test, y_pred) :
    return accuracy_score(y_test, y_pred)
    
###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def ConfusionMatrics(y_test, y_pred) :
    return confusion_matrix(y_test, y_pred)

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def Report(y_test, y_pred) :
    return classification_report(y_test, y_pred)

###################################################################################################
# Function name = ScatterPlot
# description = This function creates a scatter plot of the actual vs predicted values.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def scatterPlot(x_test, y_test, y_pred):
    plt.figure(figsize=(8,6))
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, cmap='coolwarm', alpha=0.6, label='Predicted')
    plt.title("Predicted Gender by Height and Weight")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend()
    plt.savefig("Artifacts/Weight Height/Weight_HeightLogistic_PredictionScatter.png")
    plt.close()
    
def FeatureImportance(model, x, file):
    importance = pd.Series(model.coef_[0], index=x.columns)
    file.write("Feature importance (coefficients):\n")
    file.write(str(importance.sort_values(ascending=False)) + "\n" + line + "\n")

    
###################################################################################################
# Function name = Main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 07-10-2025
###################################################################################################

def main() :
    
    try :
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Data file not found at path: {datapath}")
        
        
        #1)Load csv file
        df = ReadCSV(datapath)
        file.write("dataset loaded succesfully")
        
        Encoded_df = Mapping(df)
        
        x, y = alter(Encoded_df)
        
        x_scale = scaler(x)
        
        if sys.argv[1] == "--train" :
            
            p1 = multiprocessing.Process(target = DisplayHead, args = (df, ))
            
            p2 = multiprocessing.Process(target = DisplayShape, args = (df, ))
            
            p3 = multiprocessing.Process(target = DisplayHead, args = (df, ))
            
            p4 = multiprocessing.Process(target = DisplayCountPlot, args = (df, ))

            p5 = multiprocessing.Process(target = HeatMap, args = (Encoded_df, ))
            
            p6 = multiprocessing.Process(target = display_dimention, args = (x, y, ))
            
            p7 = multiprocessing.Process(target = PrintSize, args = (x, y, ))
            
            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            p6.start()
            
            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
            p6.join()

            """#2)display head
            DisplayHead(df, file)
            
            DisplayShape(df, file)
            
            #3)display statistical information
            DisplayStatisticalInfo(df, file)
                    
            DisplayCountPlot(Encoded_df)
            
            HeatMap(Encoded_df)
            
            display_dimention(x, y, file)
            
            PrintSize(x, y, file)
            """
            
            file.write(f"Scaled features: {x_scale}\n")

            x_train, x_test, y_train, y_test = TrainTestSplit(x_scale, y)
            file.write(f"Training set size: {x_train.shape[0]}\n")
            file.write(f"Test set size: {x_test.shape[0]}\n")
            
            PrintTrainingTestSize(x_train, x_test, y_train, y_test, file)
            
            model = fit(x_train, y_train)
            
            save_model(model, model_path)
            
        elif sys.argv[1] == "--predict" :
            x_train, x_test, y_train, y_test = TrainTestSplit(x_scale, y)
            model = load_model(model_path)
        
            y_pred = model.predict(x_test)
            file.write(f"predicted values are : ")
            file.write(f"{y_pred}\n")
                        
            accuracy = Accuracy(y_test, y_pred)
            file.write(f"accuracy is :: {accuracy}\n")

            cm = ConfusionMatrics(y_test, y_pred)
            file.write(f"confusion matrics is : {cm}\n")
            
            report = Report(y_test, y_pred)
            file.write(f"report is : {report} \n")
            
            FeatureImportance(model, x_scale, file)
            
        else :
            print("please provide valid argument --train or --predict")
            sys.exit()
            
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        exit(1)
        
    """except Exception as e :
        print(e)
        exit(1)"""
if __name__ == "__main__" :
    main()