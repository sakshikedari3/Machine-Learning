###################################################################################################
# Required Libraries
###################################################################################################s
import os
import sys
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import  SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder

###################################################################################################
# File path
###################################################################################################

line = "-" * 42
dataset = "breast-cancer-wisconsin.csv"
folder_path = "Artifacts/Breast_CancerSVM"
model_path = folder_path + "/breast_cancer_SVMmodel.joblib"
file_path = folder_path + "/" + "Breast_CancerSVM_report.txt"

###################################################################################################
# Function name = open_folder_file
# description = This function opens the folder and file for writing.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def Open_Folder_file(file_path) :
    os.makedirs(folder_path, exist_ok = True)
    if os.path.exists(file_path) :
        file = open(file_path, "a")
        
    else :
        file = open(file_path, "w")
        
    return file

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def LoadCsv() :
    return pd.read_csv(dataset)

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
# Function name = Displaycolumns
# description = This function prints the columns of the dataset.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def Displaycolumns(df, file):
    file.write(df.head().to_string())
    
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
    plt.savefig(folder_path + "/" + "heatmap.png")
    plt.close()
    #plt.show()
    
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
# Function name = DisplayShape
# description = This function prints the datatypes of the dataset.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################
    
def DisplayShape(data, file) :
    file.write(str(data.shape))
    
###################################################################################################
# Function name = scaler
# description = This function scales the features of the dataset.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def Scale(x) :
    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)
    return x_scale

###################################################################################################
# Function name = TrainTest
# description = This function splits the dataset into training and testing sets.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def TrainTest(x_scale, y) :
    a, b, c, d = train_test_split(x_scale, y, test_size = 0.2, random_state = 42)
    return a, b, c, d

###################################################################################################
# Function name = fit
# description = This function fits a logistic regression model to the training data.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def modelFit(x_train, y_train) :
    model = SVC(kernel = 'linear').fit(x_train, y_train)
    return model

###################################################################################################
# Function name = save_model
# description = This function saves the trained model to a file.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def save_model(model_fit, file, path = model_path):
    joblib.dump(model_fit, path)
    file.write(f"Model saved to {path}\n")

###################################################################################################
# Function name = load_model
# description = This function loads a trained model from a file.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def load_model(file, path = model_path):
    model = joblib.load(path)
    file.write(f"Model loaded from {path}\n")
    return model

###################################################################################################
# Function name = accuracy
# description = This function calculates the accuracy of the model.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def accuracy(y_test, y_pred) :
    return accuracy_score(y_test, y_pred)

###################################################################################################
# Function name = classification
# description = This function return the classification report.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def classification(prediction, y_test) :
    return classification_report(prediction, y_test)

###################################################################################################
# Function name = confusion
# description = This function returns the confusion matrix.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def confusionMatrix(y_test, y_pred) :
    return confusion_matrix(y_test, y_pred)

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
    plt.savefig(folder_path + "/confusion_matrix.png")
    plt.close()

###################################################################################################
# Function name = feature_importance
# description = This function displays the feature importance.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def feature_importance(model, feature_names, file) :
    feature_names = list(feature_names)
    importance = np.array(model.feature_importances_).flatten()

    file.write(f"Number of feature names: {len(feature_names)}\n")
    file.write(f"Number of importances: {len(model.feature_importances_)}\n")

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
    plt.savefig(folder_path + "/feature_importance.png")
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
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 22-9-2025
###################################################################################################

def main() :
    try :
        
        file = Open_Folder_file(file_path)
        
        df = LoadCsv()
        
        df = encoding(df)
        
        x, y = alter(df)
                    
        x_scale = Scale(x)
        
        if sys.argv[1] == '--train' :
    
            describe(df, file)
            
            Displaycolumns(df, file)
            
            datatypes(df, file)

            # 8)Display heatmap of the dataset
            heatmap(df)
                                    
            file.write(f"features are :\n{x}")
            file.write(f"target is :\n{y}")
            file.write("total records in data set : " + str(x.shape) + "\n")
            file.write("total records in data set : " + str(len(x)) + "\n")
            file.write("total records in data set : " + str(y.shape) + "\n")
            file.write("total records in data set : " + str(len(y)) + "\n")
            
            DisplayShape(df, file)
                                    
            x_train, x_test, y_train, y_test = TrainTest(x_scale, y)
            
            file.write("dimentions of traing data set : " + str(len(x_train)) + "\n")
            file.write("dimentions of traing data set : " + str(len(x_test)) + "\n")
            file.write("dimentions of traing data set : " + str(len(y_train)) + "\n")
            file.write("dimentions of traing data set : " + str(len(y_test)) + "\n")

            Model = modelFit(x_train, y_train)
            
            save_model(Model, file)
            
            file.write("Model training completed successfully.\n")  
            
            accuracy_Training = accuracy(y_train, Model.predict(x_train))
            file.write(f"training accuracy is : {accuracy_Training}\n")
            
        elif sys.argv[1] == '--test' :
            
            model = load_model(file)
    
            x_train, x_test, y_train, y_test = TrainTest(x_scale, y)
            
            prediction = model.predict(x_test)
            
            accuracy_m = accuracy(y_test, prediction)
            file.write(f"Testing accuracy is : {accuracy_m}\n")

            # 17) generate classification report and file.write report
            classif = classification(prediction, y_test)
            file.write(f"classification report is :\n{classif}\n")

            # 18) generate confusion matrix and display that matrix
            con_mat = confusionMatrix(prediction, y_test)
            file.write(f"confusion matrix is :\n{con_mat}\n")

            # 22) display auc score
            AUC_score = auc_score(y_test, prediction)
            file.write(f"AUC Score: {AUC_score:.4f}\n")

    except FileNotFoundError as e :
        print("select correct argument")

###################################################################################################
# application starter
###################################################################################################


if __name__ == "__main__" :
    main()