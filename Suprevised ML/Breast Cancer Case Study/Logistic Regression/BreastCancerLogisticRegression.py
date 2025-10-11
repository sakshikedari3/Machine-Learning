###################################################################################################
# Required Libraries
###################################################################################################

import os
import sys
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, 
    LabelEncoder)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report,
    roc_curve)

###################################################################################################
# File path
###################################################################################################

line = "-" * 42
dataset = "breast-cancer-wisconsin.csv"
model_path = "breast_cancer_model_Logistic.joblib"
folder_path = "Artifacts/Breast_Cancer"
file_path = folder_path + "/" + "Breast_Cancer_report_logistic.txt"

###################################################################################################
# Dataset Headers
###################################################################################################
headers = ['CodeNumber','ClumpThickness','UniformityCellSize','UniformityCellShape','MarginalAdhesion','SingleEpithelialCellSize','BareNuclei','BlandChromatin','NormalNucleoli','Mitoses','CancerType']

###################################################################################################
# Function name = open_folder_file
# description = This function opens the folder and file for writing.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def Open_Folder_file(file_path) :
    os.makedirs(folder_path, exist_ok = True)
    file = open(file_path, 'w')
    return file
###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def read_csv(datapath) :
    return pd.read_csv(datapath)

###################################################################################################
# Function name = describe
# description = This function print the description of the dataset.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def describe(datapath, file) :
    file.write(line)
    file.write(datapath.describe().to_string())
    
###################################################################################################
# Function name = columns
# description = This function prints the columns of the dataset.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def columns(datapath, file) :
    file.write(line)
    file.write(datapath.columns.to_series().to_string())

###################################################################################################
# Function name = datatypes
# description = This function prints the datatypes of the dataset.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def datatypes(datapath, file) :
    file.write(line)
    file.write(datapath.dtypes.to_string())

###################################################################################################
# Function name = encoding
# description = This function encodes categorical features in the dataset and returns the modified DataFrame.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################
    
def encoding(datapath) :
        
    for col in datapath.select_dtypes(include='object') :
        datapath[col] = LabelEncoder().fit_transform(datapath[col])
        
    return datapath

###################################################################################################
# Function name = heatmap
# description = This function show heat map of dataset
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def heatmap(datapath) :
    sns.heatmap(datapath.corr(), annot = True, cmap = "Purples")
    plt.title("heat map for breast cancer")
    plt.savefig(folder_path + "/" + "logisticHeatmap.png")
    plt.close()
    #plt.show()
    
def fill_empty(df) :
    df.replace('?', np.nan, inplace=True)
    df['BareNuclei'] = pd.to_numeric(df['BareNuclei'], errors='coerce')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df
    
###################################################################################################
# Function name = alter
# description = This function alters the dataset by dropping unnecessary columns.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################
    
def alter(datapath) :
    a = datapath.drop(columns=['CodeNumber','CancerType'])
    b = datapath['CancerType']
    return a, b

###################################################################################################
# Function name = scaler
# description = This function scales the features of the dataset.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def scaler(x):
    data = StandardScaler()
    return data.fit_transform(x)

###################################################################################################
# Function name = trainModel
# description = This function splits the dataset into training and testing sets.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def trainModel(x, y, test, random) :
    a, b, c, d = train_test_split(x, y, test_size=test, random_state=random)
    return a, b, c, d

###################################################################################################
# Function name = fit
# description = This function fits a logistic regression model to the training data.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def fit(x_train, y_train) :
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

###################################################################################################
# Function name = save_model
# description = This function saves the trained model to a file.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def save_model(model_fit, path = model_path):
    joblib.dump(model_fit, path)
    print(f"Model saved to {path}\n")

###################################################################################################
# Function name = load_model
# description = This function loads a trained model from a file.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def load_model():
    model = joblib.load(model_path)
    return model

###################################################################################################
# Function name = accuracy
# description = This function calculates the accuracy of the model.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def accuracy(prediction, y_test) :
    return accuracy_score(y_test, prediction)

###################################################################################################
# Function name = classification
# description = This function return the classification report.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def classification(prediction, y_test) :
    return classification_report(y_test, prediction)

###################################################################################################
# Function name = confusion
# description = This function returns the confusion matrix.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def confusion(prediction, y_test) :
    return confusion_matrix(y_test, prediction) 

###################################################################################################
# Function name = matrixDisplay
# description = This function displays the confusion matrix.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def matrixDisplay(matrix, y_test):
    cmdk = ConfusionMatrixDisplay(matrix, display_labels=np.unique(y_test))
    cmdk.plot(cmap='magma')
    plt.title("confusion matrix")
    plt.savefig("Artifacts/Breast_cancer/logisticConfusion_matrix.png")
    plt.close()

###################################################################################################
# Function name = pair_plot
# description = This function displays the pair plot.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def pair_plot(datapath) :
    df = read_csv(datapath)
    sns.pairplot(df)
    plt.title("Pair Plot")
    plt.savefig("Artifacts/Breast_cancer/logisticPair_plot.png")
    plt.close()

###################################################################################################
# Function name = feature_importance
# description = This function displays the feature importance.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def feature_importance(model, feature_names, file) :
    feature_names = list(feature_names)
    importance = np.abs(model.coef_[0])

    file.write(f"Number of feature names: {len(feature_names)}\n")
    file.write(f"Number of importances: {len(importance)}\n")
    
    if len(feature_names) != len(importance):
        raise ValueError("Length of feature_names and feature_importances_ must match.")

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='skyblue')
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("Artifacts/Breast_cancer/logisticFeature_importance.png")
    plt.close()

###################################################################################################
# Function name = auc_score
# description = This function calculates the AUC score.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def auc_score(y_test, prediction):
    auc = roc_auc_score(y_test, prediction)
    return auc

###################################################################################################
# Function name = roc_graph
# description = This function displays the ROC curve.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def roc_graph(y_test, prediction):
    lr, tlr, _ = roc_curve(y_test, prediction[:1])

    plt.plot(lr, tlr, color='blue')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of logistic regression')
    plt.grid()
    plt.savefig("Artifacts/Breast_cancer/logisticRoc_curve.png")
    plt.close()

###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def main() :
    try :
        file = Open_Folder_file(file_path)

            # 2)read dataset
        df = read_csv(dataset)
        
        df = fill_empty(df)
        
        df = encoding(df)
        
        x, y = alter(df)
        
        x_scale = scaler(x)

        if sys.argv[1] == '--train' :
            
            # 4)Describe the dataset information
            describe(df, file)

            # 5)Display information of column
            columns(df, file)

            # 6)display datatypes of columns
            datatypes(df, file)

            # 8)Display heatmap of the dataset
            heatmap(df)
                        
            file.write(f"features are :\n{x}")
            file.write(f"target is :\n{y}")

            # 11)Train the model and print its size
            x_train, x_test, y_train, y_test = trainModel(x_scale, y, 0.2, 42)

            file.write(f"size of x_train : {x_train.shape}\n")
            file.write(f"size of x_test : {x_test.shape}\n")
            file.write(f"size of y_train : {y_train.shape}\n")
            file.write(f"size of y_test : {y_test.shape}\n")

            # 12)Fit the model in algorithm
            model_fit = fit(x_train, y_train)

            # 13)save the model
            save_model(model_fit, file)
            
            file.write("Model training completed successfully.\n")  
            
            accuracy_Training = accuracy(y_train, model_fit.predict(x_train))
            file.write(f"training accuracy is : {accuracy_Training}\n")
            
        elif sys.argv[1] == '--test' :

            # 14) load the model
            model = load_model()
            
            x_train, x_test, y_train, y_test = trainModel(x_scale, y, 0.2, 42)

            # 15) make predictions
            prediction = model.predict(x_test)        

            accuracy_testing = accuracy(prediction, y_test)
            file.write(f"Testing accuracy is : {accuracy_testing}\n")

            # 17) generate classification report and print report
            classif = classification(prediction, y_test)
            file.write(f"classification report is :\n{classif}\n")

            # 18) generate confusion matrix and display that matrix
            con_mat = confusion(prediction, y_test)
            file.write(f"confusion matrix is :\n{con_mat}\n")

            # 19) display confusion matrix in visual format
            matrixDisplay(con_mat, y_test)

            # 20) display pair plot
            pair_plot(dataset)

            # 21) display feature importance
            feature_importance(model, x.columns, file)

            # 22) display auc score
            AUC_score = auc_score(y_test, prediction)
            file.write(f"AUC Score: {AUC_score:.4f}\n")

            # 23) display ROC curve
            roc_graph(y_test, prediction)
            
    except FileNotFoundError as e :
        print("select correct argument")
        print("--train or --test")

###################################################################################################
# application starter
###################################################################################################

if __name__ == "__main__" :
    main()