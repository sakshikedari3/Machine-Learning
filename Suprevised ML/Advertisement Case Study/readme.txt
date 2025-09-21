DEPENDENCIES:  
INSTALL THE FOLLOWING LIBRARIES:  
  pip install pandas, numpy, matplotlib, scikit-learn, joblib  

DATASET INFORMATION:  
   SOURCE:  
- Advertising Dataset  
- Machine-Learning/Suprevised ML/Advertisement Case Study/Advertising.csv at main · sakshikedari3/Machine-Learning

  FEATURES(3):  
1) TV  
2) Radio  
3) Newspaper  

  TARGET:  
  1) Sales  

DATA SAMPLE:  
   TV     Radio  Newspaper  Sales  
0 230.1   37.8      69.2     22.1  
1  44.5   39.3      45.1     10.4  
2  17.2   45.9      69.3      9.3  

DATA DESCRIPTION:  
- No missing values in any column  
- Correlation matrix shows TV and radio have strong correlation with sales  

WORKFLOW:  
  DATA PREPARATION:  
- Load dataset and check for missing values  
- Select features (TV, radio, newspaper) and target (sales)  

  MODEL TRAINING AND EVALUATION:  
- Train linear regression model  
- Evaluate with mean squared error, root mean squared error, R² score  

  MODEL SAVING AND LOADING:  
- Save model with joblib as advertising_model.joblib  
- Machine-Learning/Suprevised ML/Advertisement Case Study/linear_model_v1.joblib at main · sakshikedari3/Machine-Learning
- Load model for predictions without retraining  

RESULTS:  
mean square error is: 3.1740973539761033  
root mean square error is: 1.78159966153345  
R square value is: 0.899438024100912  

model coefficients are:  
TV: 0.044729517468716326  
radio: 0.18919505423437652  
newspaper: 0.0027611143413671935  
intercept c: 2.979067338122629  

MODEL STORAGE:  
- Model saved as advertising_model.joblib  

AUTHOR:  
     Sakshi Kedari
