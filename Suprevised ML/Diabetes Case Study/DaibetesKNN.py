###################################################################################################
# Required Libraries
###################################################################################################

import os
import joblib
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

###################################################################################################
# File path
###################################################################################################
line = "-" * 42
dataset = "diabetes.csv"
model_path = os.path.join("Artifacts/Diabetes", "Diabetes_KNN_model_v1.joblib")
Folder_path = "Artifacts/Diabetes"
file_path = Folder_path + "/" "Diabetes_report.txt"

###################################################################################################
# Function name = open_folder_file
# description = This function opens the folder and file for writing.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def Open_Folder_file(File_path = file_path) :
    os.makedirs(Folder_path, exist_ok = True)
    file = open(File_path, 'w')
    return file

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def read_csv(datapath) :
    df = pd.read_csv(datapath)
    return df

###################################################################################################
# Function name = Displayhead
# description = This function displays the first few rows of the DataFrame.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def DisplayHead(df, file, label="dataset sample is :"):
    file.write(f"{label}\n")
    file.write(df.head().to_string() + "\n\n")
    
###################################################################################################
# Function name = DisplayColumns
# description = This function prints the columns of the dataset.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def DisplayColumns(df, file) :
    file.write(df.columns.to_series().to_string())
    file.write("\n")
    file.write(line)
    file.write("\n")

###################################################################################################
# Function name = DisplayDescribe
# description = This function file.write the description of the dataset.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def DisplayDescribe(df, file) :
    file.write(df.describe().to_string())
    file.write("\n")
    file.write(line)
    file.write("\n")

###################################################################################################
# Function name = Alter_df
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################
    
def Alter_df(df) :
    x = df.drop(columns=['Outcome'])
    y = df['Outcome']
    return x, y

###################################################################################################
# Function name = DisplayHeat_Map
# description = This function displays the correlation matrix of the DataFrame.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def DisplayHeat_Map(df) :
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("heatmap for daibetes")
    plt.savefig("Artifacts/Diabetes/heatmap_diabetes.png")
    plt.close()
    
##################################################################################################
# Function name = histogram
# description = This function displays the histogram of the target variable.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def histogram(df) :
    plt.figure(figsize=(10,6))
    plt.hist(df['Outcome'], color = 'blue', edgecolor = 'black')
    plt.xlabel("distributation of targeted outcome")
    plt.ylabel("outcome")
    plt.title("histogram for daibetes")
    plt.savefig("Artifacts/Diabetes/histogram_diabetes.png")
    plt.close()

##################################################################################################
# Function name = box
# description = This function displays the box plot of the target variable.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def box(df) :
    plt.boxplot(df['Outcome'], patch_artist=True)
    plt.title("boxplot for daibetes")
    plt.savefig("Artifacts/Diabetes/boxplot_diabetes.png")
    plt.close()

###################################################################################################
# Function name = pair
# description = This function displays the pairplot of the DataFrame.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def pair(df) :
    sns.pairplot(df)
    plt.title("pairplot for daibetes")
    plt.savefig("Artifacts/Diabetes/pairplot_diabetes.png")
    plt.close()

###################################################################################################
# Function name = Fill_missing_values
# description = This function fills missing values in the DataFrame.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def Fill_missing_values(df) :
    df.fillna(df.mean(numeric_only=True))
    return df

###################################################################################################
# Function name = x_scaller
# description = This function scales the features using StandardScaler.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def x_scaller(x) :
    scaler = StandardScaler()
    x_scaller = scaler.fit_transform(x)
    return x_scaller

###################################################################################################
# Function name = Split_training_data
# description = This function splits the data into training and testing sets.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def Split_training_data(x_scale, y) :
    a, b, c, d = train_test_split(x_scale, y, test_size=0.2, random_state=42)
    return a, b, c, d

###################################################################################################
# Function name = fit_model
# description = This function fits the KNN model.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def fit_model(x_train, y_train) :
    model1 = KNeighborsClassifier(n_neighbors=5)
    model2 = LogisticRegression()
    model3 = DecisionTreeClassifier()
    model1.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    model3.fit(x_train, y_train)
    return model1, model2, model3

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
# Function name = Alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def Accuracy(y_test, y_pred, file) :
    accuracy = accuracy_score(y_test, y_pred)
    file.write("accuracy score is :")
    file.write(str(accuracy))
    file.write("\n")
    file.write(line)
    file.write("\n")

###################################################################################################
# Function name = Confusion_Matrix
# description = This function displays the confusion matrix.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def Confusion_Matrix(y_test, y_pred, file) :
    con = confusion_matrix(y_test, y_pred)
    file.write("confusion matrix is :\n")
    file.write(str(con))
    file.write("\n")
    file.write(line)
    file.write("\n")

###################################################################################################
# Function name = Precision
# description = This function displays the precision score.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def Precision(y_test, y_pred, file) :
    pres = precision_score(y_test, y_pred)
    file.write("precision score is :")
    file.write(str(pres))
    file.write("\n")
    file.write(line)

###################################################################################################
# Function name = Recall_score
# description = This function displays the recall score.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def Recall_score(y_test, y_pred, file) :
    rec = recall_score(y_test, y_pred)
    file.write("recall score is :")
    file.write(str(rec))
    file.write("\n")
    file.write(line)

###################################################################################################
# Function name = f1_Score
# description = This function displays the f1 score.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################
    
def f1_Score(y_test, y_pred, file) :
    f1 = f1_score(y_test, y_pred)
    file.write("f1 score is :")
    file.write(str(f1))
    file.write("\n")
    file.write(line)

###################################################################################################
# Function name = display_confusion_matrix
# description = This function displays the confusion matrix.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def display_confusion_matrix(y_test, y_pred, model_name) :
    cmd = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=np.unique(y_test))
    cmd.plot(cmap='viridis')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(f"Artifacts/Diabetes/confusion_matrix_{model_name}.png")
    plt.close()

###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def main() :
    
    try:
        
        file = Open_Folder_file()

        df = read_csv(dataset)

        if sys.argv[1] == "--train":
            
            DisplayHead(df, file)
            
            DisplayColumns(df, file)
            
            DisplayDescribe(df, file)
            
            histogram(df)
            
            box(df)
            
            pair(df)
            
            x, y = Alter_df(df)
            
            DisplayHeat_Map(df)
            
            df = Fill_missing_values(df)
            
            x_scaler = x_scaller(x)

            x_train, x_test, y_train, y_test = Split_training_data(x_scaler, y)
            
            model1, model2, model3 = fit_model(x_train, y_train)
            
            save_model(model1)
            save_model(model2)
            save_model(model3)
            
        elif sys.argv[1] == "--test":
            
            model1 = load_model(model_path)
            model2 = load_model(model_path)
            model3 = load_model(model_path)
            
            x, y = Alter_df(df)
            x_scaler = x_scaller(x)
            x_train, x_test, y_train, y_test = Split_training_data(x_scaler, y)

            y_pred1 = model1.predict(x_test)
            y_pred2 = model2.predict(x_test)
            y_pred3 = model3.predict(x_test)

            Accuracy(y_test, y_pred1, file)
            Accuracy(y_test, y_pred2, file)
            Accuracy(y_test, y_pred3, file)

            Confusion_Matrix(y_test, y_pred1, file)
            Confusion_Matrix(y_test, y_pred2, file)
            Confusion_Matrix(y_test, y_pred3, file)

            Precision(y_test, y_pred1, file)
            Precision(y_test, y_pred2, file)
            Precision(y_test, y_pred3, file)

            Recall_score(y_test, y_pred1, file)
            Recall_score(y_test, y_pred2, file)
            Recall_score(y_test, y_pred3, file)

            f1_Score(y_test, y_pred1, file)
            f1_Score(y_test, y_pred2, file)
            f1_Score(y_test, y_pred3, file)

            display_confusion_matrix(y_test, y_pred1, "KNN Classifier")
            display_confusion_matrix(y_test, y_pred2, "Logistic Regression")
            display_confusion_matrix(y_test, y_pred3, "Decision Tree Classifier")
            
        else :
            print("Invalid argument. Use --train or --test.")
            sys.exit(1)
                    
    except IndexError:
        print("Please provide an argument: --train or --test")
        
    #except Exception as e:
        #print(f"An error occurred: {e}")

if __name__ == "__main__" :
    main()