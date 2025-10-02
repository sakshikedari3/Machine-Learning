###################################################################################################
# Required Libraries
###################################################################################################s
import pandas as pd
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import countplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

###################################################################################################
# File Path and Model Path
###################################################################################################s

line = "*" * 44
datapath = "MarvellousTitanicDataset.csv"
model_path = "MarvellousTitanicDataset.joblib"
folder_path = "Artifacts/TitanicLogisticRegression"
file_path = folder_path + "/TitanicLogisticReport.txt."

###################################################################################################
# Function name = File_open
# description = This function opens a file and returns a file object.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################

def File_open() :
    os.makedirs(folder_path, exist_ok=True)
    file = open(file_path, "w")
    return file

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def ReadCSV(DATAPATH):
    return pd.read_csv(DATAPATH)

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def DisplayHead(df, file):
    file.write(f"{df.head()}\n")
    file.write(line + "\n")

###################################################################################################
# Function name = DisplayStatisticalInfo
# description = This function displays the statistical information of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def DisplayStatisticalInfo(df, file):
    file.write("statistical information are : \n")
    file.write(line + "\n")
    file.write(f"{df.describe()}\n") # use for prepare and manipulate
    file.write(line + "\n")
    #file.write(f"{df.info()}\n") # use for prepare and manipulate
    file.write(line + "\n")
    
###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def DisplayDimension(df, file):
    file.write(f"dimention of dataset is : {df.shape}\n")
    file.write(line + "\n")
    
###################################################################################################
# Function name = 
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def PreprocessData(df):
    df.drop(columns = ['Passengerid','zero'], inplace = True)
    return df

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def AnalyzeAndModel(df, file):
    file.write(f"dimention of dataset is : {df.shape}\n")
    file.write(line + "\n")

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def Encoding(df):
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
    return df

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def DisplayCountPlot(df):
    plt.figure()
    target = "Survived"
    countplot(data = df, x = target).set_title("survived vs non survive")
    plt.savefig("Artifacts/TitanicLogisticRegression/survived_vs_non_survive.png")
    plt.close()

    plt.figure()
    target = "Survived"
    countplot(data = df, x = target, hue = 'Sex').set_title("based on gender")
    plt.savefig("Artifacts/TitanicLogisticRegression/survived_vs_gender.png")
    plt.close()

    plt.figure()
    target = "Survived"
    countplot(data = df, x = target, hue = 'Pclass').set_title("based on pclass")
    plt.savefig("Artifacts/TitanicLogisticRegression/survived_vs_pclass.png")
    plt.close()

    plt.figure()
    df['Age'].plot.hist().set_title("age report")
    plt.savefig("Artifacts/TitanicLogisticRegression/age_report.png")
    plt.close()

    plt.figure()
    df['Fare'].plot.hist().set_title("fare report")
    plt.savefig("Artifacts/TitanicLogisticRegression/fare_report.png")
    plt.close()

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def DisplayHeatMap(df):
    plt.figure(figsize=(10,6))

    sns.heatmap(df.corr(), annot=True, cmap = "coolwarm")

    plt.title("feature correlation heatmap")
    plt.savefig("Artifacts/TitanicLogisticRegression/feature_correlation_heatmap.png")
    plt.close()
    
###################################################################################################
# Function name = alter
# description = This function alters the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def ModelBuilding(df):
    x = df.drop(columns = ['Survived'])
    y = df['Survived']
    return x, y

###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def display_dimention(x, y, file):
    file.write("dimentation of feature : ")
    file.write(str(x.shape))
    file.write("dimentation of label : ")
    file.write(str(y.shape))
    file.write(line + "\n")

###################################################################################################
# Function name = scaler
# description = This function scales the features using StandardScaler.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def scaler(x) :
    scaller = StandardScaler()
    x_scale = scaller.fit_transform(x)
    return x_scale

###################################################################################################
# Function name = printSize
# description = This function prints the size of the independent and dependent variables.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def PrintSize(x, y, file) :
    file.write("independent variables are : Head Size\n")
    file.write("dependent variables are : Brain Weight\n")
    file.write("total records in data set : {}\n")
    file.write(format(x.shape))
    file.write(line)
    file.write("total records in data set : {}\n")
    file.write(format(len(x)))
    file.write(line)
    file.write("total records in data set : {}\n")
    file.write(format(y.shape))
    file.write(line)
    file.write("total records in data set : {}\n")
    file.write(format(len(y)))
    file.write(line + "\n")

###################################################################################################
# Function name = printSize
# description = This function prints the size of the independent and dependent variables.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def TrainTestSplit(x_scale, y) :
    a, b, c, d = train_test_split(x_scale, y, test_size=0.2, random_state=42)
    return a, b, c, d

###################################################################################################
# Function name = PrintTrainingTestSize
# description = This function prints the size of the training and testing datasets.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def PrintTrainingTestSize(x_train, x_test, y_train, y_test, file):
    file.write("dimentions of traing data set : {}\n")
    file.write(format(len(x_train)))
    file.write(line)
    file.write("dimentions of traing data set : {}\n")
    file.write(format(len(x_test)))
    file.write(line)
    file.write("dimentions of traing data set : {}\n")
    file.write(format(len(y_train)))
    file.write(line)
    file.write("dimentions of traing data set : {}\n")
    file.write(format(len(y_test)))
    file.write(line + "\n")
    
###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def FitModel(x_train, y_train) :

    model = LogisticRegression()

    model.fit(x_train, y_train)
    
    return model

###################################################################################################
# Function name = save_model
# description = This function saves the trained model using joblib.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
###################################################################################################
# Function name = load_model
# description = This function loads the trained model using joblib.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def load_model(model_path):
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)


###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def Accuracy(y_test, y_pred) :
    return accuracy_score(y_test, y_pred)
    
###################################################################################################
# Function name = DisplayHead
# description = This function displays the first few records of the DataFrame.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def ConfusionMatrics(y_test, y_pred) :
    return confusion_matrix(y_test, y_pred)

###################################################################################################
# Function name = Main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 16-9-2025
###################################################################################################
def main() :
    
    try :
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Data file not found at path: {datapath}")
    
        file = File_open()
        
        #1)Load csv file
        df = ReadCSV(datapath)
        file.write("dataset loaded succesfully")
        
        Encoded_df = Encoding(df)
        
        x, y = ModelBuilding(Encoded_df)
        
        x_scale = scaler(x)
        
        if sys.argv[1] == "--train" :
            
        
            #2)display head
            DisplayHead(df, file)
            
            #3)display statistical information
            DisplayStatisticalInfo(df, file)
            
            #4)Display dimensions
            DisplayDimension(df, file)
            
            #5)
            df = PreprocessData(df)
            
            AnalyzeAndModel(df, file)
            
            
            DisplayCountPlot(Encoded_df)
            
            DisplayHeatMap(Encoded_df)
            
            
            display_dimention(x, y, file)
            


            file.write(f"Scaled features: {x_scale}\n")

            x_train, x_test, y_train, y_test = TrainTestSplit(x_scale, y)
            file.write(f"Training set size: {x_train.shape[0]}\n")
            file.write(f"Test set size: {x_test.shape[0]}\n")
            
            PrintTrainingTestSize(x_train, x_test, y_train, y_test, file)
            
            model = FitModel(x_train, y_train)
            
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
            
        else :
            print("please provide valid argument --train or --predict")
            sys.exit()
            
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        exit(1)
        
    except Exception as e :
        print(e)
        exit(1)

###################################################################################################
# application starter
###################################################################################################
if __name__ == "__main__" :
    main()