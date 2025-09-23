###################################################################################################
# Required Libraries
###################################################################################################
import os
import sys
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

###################################################################################################
# File path
###################################################################################################
line = "-" * 42
dataset = "WinePredictor.csv"
model_path = os.path.join("Artifacts/Wine_Predictor", "WinePredictor_KNN_model_v1.joblib")
Folder_path = "Artifacts/Wine_Predictor"
file_path = Folder_path + "/" "Wine_Predictor_report.txt"

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
    plt.savefig("Artifacts/Wine_Predictor/Correlation_plot.png")
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
    plt.savefig("Artifacts/Wine_Predictor/Pairplot_plot.png")
    plt.close()

###################################################################################################
# Function name = Alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def alter(df) :
    x = df.drop(columns=['Class','Flavanoids','OD280/OD315 of diluted wines','Hue','Alcalinity of ash'])
    y = df['Class']
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

def save_model(model, x_test, y_test, model_path = model_path) :
    joblib.dump(model, model_path)
    joblib.dump(x_test, "Artifacts/Wine_Predictor/X_test.joblib")
    joblib.dump(y_test, "Artifacts/Wine_Predictor/Y_test.joblib")

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
    model = KNeighborsClassifier(n_neighbors=50)
    model.fit(x_train, y_train)
    return model

###################################################################################################
# Function name = Alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def accuracy_test(y_test, y_pred):
    accuracy_data = []
    k_range = range(1, 23)
    for i in k_range:
        
        accuracy = accuracy_score(y_test, y_pred)

        accuracy_data.append(accuracy)
        
    return accuracy_data, k_range

###################################################################################################
# Function name = Alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def DisplayHeatMap(x) :
        
    plt.figure(figsize=(10,6))
    
    sns.heatmap(x.corr(), annot = True, cmap = 'YlGnBu')
    
    plt.title("co-relation between columns")

    plt.savefig("Artifacts/Wine_Predictor/HeatMap.png")
    
    plt.close()
    
def plot(accuracy, k_range) :
    plt.plot(accuracy, k_range, color = 'blue')
    plt.xlabel('Accuracy')
    plt.ylabel('K value')
    plt.title('K value vs Accuracy')
    plt.savefig("Artifacts/Wine_Predictor/K_value_vs_Accuracy.png")
    plt.close()

###################################################################################################
# Function name = Alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 23-9-2025
###################################################################################################

def b_accuracy(accuracy_data) :
    return max(accuracy_data)

###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 21-9-2025
###################################################################################################

def main() :
    try :
        file = Open_Folder_file()
        
        df = read_csv(dataset)
        
        if sys.argv[1] == '--train' :
            
            Displayhead(df, file)
            
            describe(df, file)
            
            columns(df, file)
            
            datatypes(df, file)
            
            x, y = alter(df)
            
            x_train, x_test, y_train, y_test = train_model(x, y)
            
            model = fit_model(x_train, y_train)
            
            DisplayHeatMap(x)
            
            DisplayPairplot(df)
            
            DisplayCorrelation(df, file)

            save_model(model, x_test, y_test)
            
            file.write("model is trained and saved \n")
            
        elif sys.argv[1] == '--predict' :
            
            x, y = alter(df)

            x_train, x_test, y_train, y_test = train_model(x, y)
            
            model = fit_model(x_train, y_train)

            model = load_model()

            file.write("model is loaded \n")
    
            y_pred = model.predict(x_test)
            
            accuracy, k_range = accuracy_test(y_test, y_pred)
                    
            best_accuracy = b_accuracy(accuracy)
            
            plot(accuracy, k_range)
            
            file.write("best accuracy is : ")
            file.write(str(best_accuracy))

        else :
            print("Please provide valid argument : --train or --predict")
            return
            
    except Exception as e :
        print(e)
        print("Please provide valid argument : --train or --predict")

if __name__ == "__main__" :
    main()