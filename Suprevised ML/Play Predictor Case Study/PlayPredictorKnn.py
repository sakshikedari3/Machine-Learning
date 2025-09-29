###################################################################################################
# Required Libraries
###################################################################################################
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib

###################################################################################################
# File path
###################################################################################################
line = "-" * 42
dataset = "PlayPredictor.csv"
model_path = "Artifacts/PlayPredictor/playPredictor_model.joblib"
folder_path = "Artifacts/PlayPredictor"
file_path = folder_path + "/" + "PlayPredictor.txt"

###################################################################################################
# Function name = Read_file
# description = This function opens a file and returns a DataFrame.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################

def read_file(file_path):
    os.makedirs(folder_path, exist_ok = True)
    fobj = open(file_path, 'w')
    return fobj

###################################################################################################
# Function name = ReadCsv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def ReadCsv(datapath) :
    return pd.read_csv(datapath)
    
###################################################################################################
# Function name = display_head
# description = This function displays the head of a DataFrame.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def display_head(datapath, file) :
    file.write(datapath.head().to_string())
    file.write(line)

###################################################################################################
# Function name = Describe
# description = This function displays the description of a DataFrame.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def Describe(df, file) :
    file.write(df.describe().to_string())
    file.write(line)
    
###################################################################################################
# Function name = Scaller
# description = This function scales the features of a DataFrame.
# author = sakshi kedari    
# date = 17-8-2025
###################################################################################################

def Scaler(df) :
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

"""###################################################################################################
# Function name = Encoder
# description = This function encodes categorical variables into numerical format using Label Encoding.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################

def Encoder(df) :
    label_encoder = LabelEncoder()
    df['Whether'] = label_encoder.fit_transform(df['Whether'])
    df['Temperature'] = label_encoder.fit_transform(df['Temperature'])
    df['Play'] = label_encoder.fit_transform(df['Play'])
    return df
"""

###################################################################################################
# Function name = DisplayColumns
# description = This function displays the columns of a DataFrame.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def DisplayColumns(df, file) :
    file.write(df.columns.to_series().to_string())
    file.write(line)

###################################################################################################
# Function name = DisplayDatatypes
# description = This function displays the datatypes of a DataFrame.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def DisplayDatatypes(df, file) :
    file.write(df.dtypes.to_string())
    file.write(line)

###################################################################################################
# Function name = HeatMap
# description = This function displays the heatmap of correlations in a DataFrame.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def HeatMap(df) :
    plt.figure(figsize = (10, 6))
    sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm', fmt = '.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('Artifacts/PlayPredictor/heatmap.png')
    plt.close()

###################################################################################################
# Function name = Encoding
# description = This function encodes categorical variables into numerical format.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def Encoding(df):
    df['Whether'] = df['Whether'].map({'Sunny' : 0, 'Overcast' : 1, 'Rainy' : 2})
    df['Temperature'] = df['Temperature'].map({'Hot' : 0, 'Mild' : 1, 'Cool' : 2})
    df['Play'] = df['Play'].map({'Yes' : 1, 'No' : 0})
    return df

###################################################################################################
# Function name = alter
# description = This function alters the DataFrame for model training.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def alter(df) :
    x = df.drop(columns=['Play'])
    y = df['Play'] 
    return x, y

###################################################################################################
# Function name = model_training
# description = This function splits the data into training and testing sets.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def model_training(x, y) :
    a, b, c, d = train_test_split(x, y, test_size=0.2, random_state=42) 
    return a, b, c, d

###################################################################################################
# Function name = fit_model
# description = This function fits the KNN model.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def fit_model(x_train, y_train) :

    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train, y_train)
    return model

###################################################################################################
# Function name = SaveModel
# description = This function saves the trained model to a file.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def SaveModel(model, model_path) :
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

###################################################################################################
# Function name = LoadModel
# description = This function loads a trained model from a file.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def LoadModel(model_path) :
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

###################################################################################################
# Function name = accuracy
# description = This function calculates the accuracy of the model.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def accuracy(y_test, y_pred) :
    return accuracy_score(y_test, y_pred)

###################################################################################################
# Function name = confusion
# description = This function calculates the confusion matrix of the model.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def confusion(y_test, y_pred) :
    return confusion_matrix(y_test, y_pred)

###################################################################################################
# Function name = DisplayConfusionMatrix
# description = This function displays the confusion matrix of the model.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def DisplayConfusionMatrix(cm, model) :
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('Artifacts/PlayPredictor/confusion_matrix.png')
    plt.close()

###################################################################################################
# Function name = classification
# description = This function calculates the classification report of the model.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def classification(y_test, y_pred, file) :
    file.write(classification_report(y_test, y_pred, zero_division=0))
    file.write(line)
    
###################################################################################################
# Function name = plot_graph
# description = This function plots the graph of the model's performance.
# author = sakshi kedari
# date = 17-8-2025
###################################################################################################
def plot_graph(x, y) :
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot')
    plt.savefig('Artifacts/PlayPredictor/scatter_plot.png')
    plt.close()

###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 13-8-2025
###################################################################################################
def main() :

    
    try :
        if len(sys.argv) != 2 or sys.argv[1] not in ["--train", "--test"]:
            raise ValueError("Invalid arguments. Use --train or --test.")
    
        #1) Read CSV file
        df = ReadCsv(dataset)
    
        
        df = Encoding(df)
        

        #9) Alter
        x, y = alter(df)
    
        
        file = read_file(file_path)
    
        
        if sys.argv[1] == "--train":
        
            #2) Display head
            display_head(df, file)
        
            
            #3) Describe
            Describe(df, file)
        
            df = Scaler(df)
            
            #4) Display Info
            
            #5) Display Columns
            DisplayColumns(df, file)
        
            
            #6) Display Datatypes
            DisplayDatatypes(df, file)
            
            
            #7) Encoding
            df = Encoding(df)
            

            #8) HeatMap
            HeatMap(df)
            

            #10) Model Training
            x_train, x_test, y_train, y_test = model_training(x, y)
            
            
            #11) Fit Model
            model = fit_model(x_train, y_train)
            

            #12) Save Model
            SaveModel(model, model_path)
            
            
        elif sys.argv[1] == "--test":
            
            x_train, x_test, y_train, y_test = model_training(x, y)
            
        
            #13) Load Model
            model = LoadModel(model_path)
            
            
            #14) Prediction
            y_pred = model.predict(x_test)
            
            Accuracy = accuracy(y_test, y_pred)
            file.write(f"accuracy is : {Accuracy}\n")
            file.write(line)
            
            
            #16) confusion matrix
            cm = confusion(y_test, y_pred)
            file.write(f"confusion matrix is :\n{cm}\n")
            file.write(line)
            
            
            #17) Display confusion matrix
            DisplayConfusionMatrix(cm, model)
            
            
            #18) classification report
            classification(y_test, y_pred, file)
            
            
            #19) plot graph
            plot_graph(x_test.iloc[:, 0], y_test)
            
        
    except Exception as E:
        print("Error : ",E)
        sys.exit(1)
        
###################################################################################################
# application starter
###################################################################################################
if __name__ == "__main__" :
    main()