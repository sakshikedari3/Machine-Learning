
# 📞 Telco Customer Churn Prediction Case Study

## 📘 Overview  
This project predicts customer churn using a Random Forest Classifier trained on the Telco Customer dataset. It demonstrates a reproducible machine learning workflow with modular preprocessing, training, evaluation, and artifact logging.

---

## 🧩 Problem Statement  
Identify customers likely to churn based on service usage and demographic features. This helps telecom companies improve retention strategies and reduce revenue loss.

---

## 📊 Dataset  
**File:** `TelcoCustomer.csv`  
**Features:**  
- Demographics: `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `gender`  
- Services: `PhoneService`, `InternetService`, `OnlineSecurity`, `StreamingTV`, etc.  
- Financials: `MonthlyCharges`, `TotalCharges`, `PaymentMethod`  
- Target: `Churn`  
**Size:** 7043 records

---

## ⚙️ Features

### 🔄 Data Preprocessing  
- Dropped non-informative columns: `customerID`, `gender`  
- Label encoding for categorical features  
- Feature scaling using `StandardScaler`  
- Train-test split with reproducibility

### 📈 Visualization  
- Heatmap of feature correlations  
- Pairplot for feature relationships  
- Feature importance plot from Random Forest  
- Confusion matrix for prediction results

### 🤖 Model  
- Algorithm: Random Forest Classifier  
- Hyperparameters: `n_estimators=150`, `max_depth=7`, `random_state=42`  
- Evaluation: Accuracy, Precision, Confusion Matrix  
- Artifacts: Model and plots saved automatically

---

## 🧪 Technical Implementation  
- Modular Python functions for each step  
- Train-test split using `train_test_split`  
- Model saving and loading with `joblib`  
- Command-line interface using `--train` and `--predict` flags

---

## 🚀 Usage

### 📦 Prerequisites  
Install required dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ Running the Application  
```bash
python telco_churn_pipeline.py --train     # Train the model  
python telco_churn_pipeline.py --predict   # Evaluate the model  
```

### 🧾 Command Line Arguments  
- `--train`: Trains the model and saves artifacts  
- `--predict`: Loads the model and evaluates performance

---

## 📤 Output  
The application generates:

- **Model Performance Metrics:**  
  - Accuracy  
  - Precision  
  - Confusion Matrix  
- **Visualizations:** Saved in `Artifacts/TelcoCustomber/`  
  - `heatmap.png`  
  - `pairplot.png`  
  - `FeatureImportance.png`  
- **Trained Model:**  
  - `Telco_Customber_RandomForest.joblib`  
- **Logs:** Written to `TelcoCustomerReport.txt`

---

## 📊 Sample Report Output  
```
accuracy score is : 0.79  
precision score is : 0.68  
confusion matrix is :
[[78 21]
 [18 37]]
```

---

## 🔍 Key Insights  
- Random Forest provides interpretable feature importance  
- Label encoding and scaling improve model performance  
- Modular design supports reproducibility and debugging

---

## 🏢 Business Applications  
This model can be adapted for:

- Telecom Customer Retention  
- Subscription-based Service Analytics  
- Business Intelligence Dashboards  
- Data Science Training Projects

---

## 📁 Dataset Features Description  
- `tenure`: Number of months the customer has stayed  
- `MonthlyCharges`: Monthly billing amount  
- `InternetService`, `StreamingTV`, etc.: Service usage indicators  
- `Churn`: Target variable (Yes/No)

---

## 📂 File Structure  
```
TelcoCustomer/
├── telco_churn_pipeline.py
├── TelcoCustomer.csv
├── requirements.txt
├── README.md
└── Artifacts/
    └── TelcoCustomber/
        ├── heatmap.png
        ├── pairplot.png
        ├── FeatureImportance.png
        ├── Telco_Customber_RandomForest.joblib
        └── TelcoCustomerReport.txt
```

---

## 📦 Dependencies  
```
pandas >= 2.1.0  
numpy >= 1.25.0  
matplotlib >= 3.8.0  
seaborn >= 0.12.2  
scikit-learn >= 1.3.0  
joblib >= 1.3.2  
```

---

## 👩‍💻 Author  
**Sakshi Kedari**  
Date: 05/10/2025

