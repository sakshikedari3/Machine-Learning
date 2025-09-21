DEPENDENCIES:  
INSTALL THE FOLLOWING LIBRARIES:  
  pip install pandas, seaborn, joblib, numpy, matplotlib, scikit-learn  

DATASET INFORMATION:  
  SOURCE:  
  - Advertising Dataset (Advertising.csv)  

  FEATURES(4):  
  1) Unnamed: 0 (Index)  
  2) TV  
  3) Radio  
  4) Newspaper  

  TARGET:  
  1) Sales  

DATA SAMPLE:  
   Unnamed: 0     TV  radio  newspaper  sales  
0           1  230.1   37.8       69.2   22.1  
1           2   44.5   39.3       45.1   10.4  
2           3   17.2   45.9       69.3    9.3  
3           4  151.5   41.3       58.5   18.5  
4           5  180.8   10.8       58.4   12.9  

DATA DESCRIPTION:  
- No missing values in any column  
- Features after dropping 'Unnamed: 0' used for modeling  
- Correlation matrix shows TV and radio have strong correlation with Sales  

WORKFLOW:  
  DATA PREPARATION:  
  - Load CSV file, remove 'Unnamed: 0' column  
  - Display dataset sample, description, columns, datatypes  
  - Check missing values (0 for all columns)  
  - Show correlation matrix and pairplot  

  TRAIN TEST SPLIT:  
  - Split data into 80% training and 20% testing sets  

  MODEL TRAINING AND EVALUATION:  
  - Train Linear Regression model  
  - Evaluate using mean squared error, root mean squared error, R² score  
  - Save trained model to Artifacts/Advertisement/linear_model_v1.joblib  
  - Output features and target samples during training  

  MODEL SAVING AND LOADING:  
  - Load saved model for predictions without retraining  

RUNNING THE PROJECT:  
  - Train model:  
    python script.py --train  
  - Predict / baseline:  
    python script.py --predict  

EXPECTED OUTPUT (after training):  
  - Model saved at Artifacts/Advertisement/linear_model_v1.joblib  
  - Model training completed  

VISUALIZATIONS:  
  - Correlation heatmap (saved as Correlation_plot.png)  
  - Pairplot of features (saved as Pairplot_plot.png)  
  - Scatter plot of actual vs predicted sales (saved as prediction_plot.png)  

MODEL PERFORMANCE (example):  
mean square error is: 3.1740973539761033  
root mean square error is: 1.78159966153345  
R square value is: 0.899438024100912  

MODEL COEFFICIENTS:  
  TV: 0.044729517468716326  
  radio: 0.18919505423437652  
  newspaper: 0.0027611143413671935  
  intercept c: 2.979067338122629  

MODEL STORAGE:  
  - Model saved at Artifacts/Advertisement/linear_model_v1.joblib  
  - Load anytime:  
    from joblib import load  
    model = load("Artifacts/Advertisement/linear_model_v1.joblib")  

SAMPLE PREDICTION:  
  sample = X_test.iloc[]
  pred = model.predict(sample)  
  print("prediction:", pred)  # predicted sales value

AUTHOR:  
Sakshi Santosh Kedari  

date: 21/09/2025, 6 PM IST  
