###################################################################################################
# Required Libraries
###################################################################################################

import os
import sys
import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score

###################################################################################################
# File path
###################################################################################################
line = "-" * 42
dataset = "TelcoCustomer.csv"
model_path = os.path.join("Artifacts/TelcoCustomber", "Telco_Customber_RandomForest.joblib")
Folder_path = "Artifacts/TelcoCustomber"
file_path = Folder_path + "/" "TelcoCustomerReport.txt"

###################################################################################################
# Function name = open_folder_file
# description = This function opens the folder and file for writing.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def Open_Folder_file(File_path = file_path) :
    os.makedirs(Folder_path, exist_ok = True)
    file = open(File_path, 'w')
    return file

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def read_csv(datapath) :
    df = pd.read_csv(datapath)
    return df

###################################################################################################
# Function name = Displayhead
# description = This function displays the first few rows of the DataFrame.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def DisplayHead(df, file, label="dataset sample is :"):
    file.write(f"{label}\n")
    file.write(df.head().to_string() + "\n\n")
    print(df.head())
    
###################################################################################################
# Function name = DisplayColumns
# description = This function prints the columns of the dataset.
# author = sakshi kedari
# date = 05-10-2025
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
# date = 05-10-2025
###################################################################################################

def DisplayDescribe(df, file) :
    file.write(df.describe().to_string())
    file.write("\n")
    file.write(line)
    file.write("\n")
    
###################################################################################################
# Function name = DropColumn
# description = This function file.write the description of the dataset.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################
    
def DropColumn(df) :
    df.drop(columns=['customerID','gender'], axis=1, inplace=True)
    return df

###################################################################################################
# Function name = labelEncoder
# description = This function file.write the description of the dataset.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def labelEncoder(df) :
    for col in df.select_dtypes(include='object') :
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

###################################################################################################
# Function name = Alter_df
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################
    
def Alter_df(df) :
    x = df.drop('Churn', axis=1)
    y = df['Churn']
    return x, y

###################################################################################################
# Function name = DisplayHeat_Map
# description = This function displays the correlation matrix of the DataFrame.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def DisplayHeat_Map(df) :
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("heatmap for Telco Customber")
    plt.savefig("Artifacts/TelcoCustomber/heatmap.png")
    plt.close()
    
###################################################################################################
# Function name = pair
# description = This function displays the pairplot of the DataFrame.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def pair(df) :
    sns.pairplot(df)
    plt.title("pairplot for Telco customber")
    plt.savefig("Artifacts/TelcoCustomber/pairplot.png")
    plt.close()
    
###################################################################################################
# Function name = x_scaller
# description = This function scales the features using StandardScaler.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def x_scaller(x) :
    scaler = StandardScaler()
    x_scaller = scaler.fit_transform(x)
    return x_scaller

###################################################################################################
# Function name = Split_training_data
# description = This function splits the data into training and testing sets.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def Split_training_data(x_scale, y) :
    a, b, c, d = train_test_split(x_scale, y, test_size=0.2, random_state=42)
    return a, b, c, d

###################################################################################################
# Function name = fit_model
# description = This function fits the KNN model.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def fit_model(x_train, y_train) :
    model = RandomForestClassifier(n_estimators = 150, max_depth = 7, random_state = 42)
    model.fit(x_train, y_train)
    return model

###################################################################################################
# Function name = save_model    
# description = This function saves the trained model.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def save_model(model, model_path = model_path) :
    joblib.dump(model, model_path)
    
###################################################################################################
# Function name = load_model
# description = This function loads the trained model.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def load_model(model_path = model_path) :  
    model = joblib.load(model_path)
    return model

###################################################################################################
# Function name = Alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def Accuracy(y_test, y_pred, file) :
    accuracy = accuracy_score(y_test, y_pred)
    file.write("accuracy score is :")
    file.write(str(accuracy))
    file.write("\n")
    file.write(line)
    file.write("\n")
    
###################################################################################################
# Function name = DisplatFreatureImportance
# description = This function displays the feature importance.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################
    
def DisplatFreatureImportance(model, x) :
    importance = pd.Series(model.feature_importances_, index=x.columns)
    importance = importance.sort_values(ascending = False)
    #print(importance)
    importance.plot(kind = 'bar', figsize=(10, 6), title = 'feature importance')
    plt.savefig("Artifacts/TelcoCustomber/FeatureImportance.png")
    plt.close()

###################################################################################################
# Function name = Confusion_Matrix
# description = This function displays the confusion matrix.
# author = sakshi kedari
# date = 05-10-2025
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
# date = 05-10-2025
###################################################################################################

def Precision(y_test, y_pred, file) :
    pres = precision_score(y_test, y_pred)
    file.write("precision score is :")
    file.write(str(pres))
    file.write("\n")
    file.write(line)

###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 05-10-2025
###################################################################################################

def main() :
    try :
        
        file = Open_Folder_file()
            
        df = read_csv(dataset)
        
        if sys.argv[1] == "--train" :
            
            DisplayHead(df, file)
            
            DisplayDescribe(df, file)
            
            DisplayColumns(df, file)
            
            df = DropColumn(df)
            
            df = labelEncoder(df)
            
            x, y = Alter_df(df)
        
            scale_x = x_scaller(x)
            
            DisplayHeat_Map(df)
            
            pair(df)            
            
            print("type of scale_x", type(scale_x))
            
            x_train, x_test, y_train, y_test = Split_training_data(scale_x, y)
            
            model = fit_model(x_train,y_train)
            
            save_model(model)
            
            print("Model trained and saved successfully!")
            
        elif sys.argv[1] == "--predict" :
            
            x, y = Alter_df(df)
        
            scale_x = x_scaller(x)
            
            x_train, x_test, y_train, y_test = Split_training_data(scale_x, y)
            
            model = load_model()
            
            y_pred = model.predict(x_test)    
            
            Accuracy(y_pred, y_test, file)
            
            Confusion_Matrix(y_test, y_pred, file)
            
            Confusion_Matrix(y_test, y_pred, file)
            
            DisplatFreatureImportance(model, x)
            
            Precision(y_test, y_pred, file)
            
        else :
            print("select correct option!")
            print("--train or --predict")
            sys.exit(1)
            
    except Exception as e:
        print(e)
    
if __name__ == "__main__" :
    main()