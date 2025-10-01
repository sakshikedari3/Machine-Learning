# 🧠 HeadBrain Linear Regression Case Study

## 📘 Overview  
This case study explores the relationship between head size and brain weight using a simple linear regression model. It demonstrates a reproducible machine learning pipeline with modular preprocessing, training, evaluation, and artifact logging.

---

## 🧩 Problem Statement  
Predict brain weight based on head size using linear regression. This helps understand how physical measurements correlate and can be used for educational or analytical purposes.

---

## 📊 Dataset  
**File:** HeadBrain.csv  
**Features:**  
- Independent Variable: `Head Size(cm^3)`  
- Dependent Variable: `Brain Weight(grams)`  
**Size:** 237 records

---

## ⚙️ Features

### 🔄 Data Preprocessing  
- Feature selection: `Head Size(cm^3)` as input, `Brain Weight(grams)` as output  
- Feature scaling using `StandardScaler`  
- Train-test split with reproducibility

### 📈 Visualization  
- Correlation heatmap using seaborn  
- Scatter plot of predictions vs actuals

### 🤖 Model  
- Algorithm: Linear Regression  
- Evaluation: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score  
- Artifacts: Model and plots saved automatically

---

## 🧪 Technical Implementation  
- **Preprocessing:** Feature scaling with `StandardScaler`  
- **Pipeline:** Modular Python functions for each step  
- **Validation:** 80/20 train-test split with `random_state=42`  
- **Model Saving:** Using `joblib` with reproducible file paths  
- **Command-Line Interface:** Supports `--train` and `--predict` modes

---

## 🚀 Usage

### 📦 Prerequisites  
Install required dependencies:
```bash
pip install pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
```

### ▶️ Running the Application  
```bash
python HeadBrainLinearModel.py --train     # Train the model  
python HeadBrainLinearModel.py --predict   # Evaluate the model  
```

### 🧾 Command Line Arguments  
- `--train`: Trains the model and saves artifacts  
- `--predict`: Loads the model and evaluates performance  

---

## 📤 Output  
The application generates:

- **Model Performance Metrics:**  
  - Mean Square Error  
  - Root Mean Square Error  
  - R² Score  
- **Visualizations:** Saved in `Artifacts/HeadBrain/`  
  - `Heatmap.png`: Correlation heatmap  
  - `ScatterPlot.png`: Predicted vs actual brain weights  
- **Trained Model:** Saved as `HeadBrainLinearModel.joblib`  
- **Logs:** Written to `HeadBrain.txt`

---

## 📊 Model Performance  
- Mean Squared Error: Typically low due to linear relationship  
- Root Mean Squared Error: Indicates average prediction error  
- R² Score: Measures strength of correlation between head size and brain weight

---

## 🔍 Key Insights  
- Head size shows a strong linear correlation with brain weight  
- Simple regression is effective for small, clean datasets  
- Scaling improves model stability and interpretability

---

## 🏢 Business Applications  
This model can be adapted for:

- Educational Tools: Teaching regression concepts  
- Biomedical Research: Exploring physical correlations  
- Data Science Training: Demonstrating end-to-end ML pipelines

---

## 📁 Dataset Features Description  
- `Head Size(cm^3)`: Volume of the head  
- `Brain Weight(grams)`: Weight of the brain

---

## 📂 File Structure  
```
HeadBrain/
├── HeadBrainLinearModel.py
├── HeadBrain.csv
├── requirements.txt
├── README.md
└── Artifacts/
    └── HeadBrain/
        ├── Heatmap.png
        ├── ScatterPlot.png
        ├── HeadBrain.txt
        └── HeadBrainLinearModel.joblib
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
Date: 01/10/2025
