
# 🧍‍♂️🧍‍♀️ Weight-Height Gender Classification Case Study

## 📘 Overview  
This project predicts gender based on height and weight using logistic regression. It demonstrates a reproducible classification workflow with modular preprocessing, visualization, training, evaluation, and artifact logging.

---

## 🧩 Problem Statement  
Classify individuals as male or female based on their height and weight. This helps explore how physical attributes can be used for binary classification tasks.

---

## 📊 Dataset  
**File:** `weight-height.csv`  
**Features:**  
- `Height`: Height in inches  
- `Weight`: Weight in pounds  
- `Gender`: Target variable (Male/Female)  
**Size:** 10,000 records

---

## ⚙️ Features

### 🔄 Data Preprocessing  
- Manual encoding of `Gender` using `.map()`  
- Feature scaling with `StandardScaler`  
- Train-test split with reproducibility

### 📈 Visualization  
- Count plots for gender distribution  
- Histograms for height and weight  
- Heatmap of feature correlations  
- Scatter plot of predictions

### 🤖 Model  
- Algorithm: Logistic Regression  
- Evaluation: Accuracy, Confusion Matrix, Classification Report  
- Artifacts: Model and plots saved automatically

---

## 🧪 Technical Implementation  
- Modular Python functions for each step  
- Multiprocessing used for parallel EDA tasks  
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
python weight_height_logistic.py --train     # Train the model  
python weight_height_logistic.py --predict   # Evaluate the model  
```

### 🧾 Command Line Arguments  
- `--train`: Trains the model and saves artifacts  
- `--predict`: Loads the model and evaluates performance

---

## 📤 Output  
The application generates:

- **Model Performance Metrics:**  
  - Accuracy  
  - Confusion Matrix  
  - Classification Report  
- **Visualizations:** Saved in `Artifacts/Weight Height/`  
  - `Weight_HeightLogistic_HeatMap.png`  
  - `Weight_HeightLogistic_ScatterPlot.png`  
  - `Weight_HeightLogistic_PredictionScatter.png`  
- **Trained Model:**  
  - `Weight_HeightLogisticModel.joblib`  
- **Logs:** Written to `Weight_HeightLogistic.txt`

---

## 📊 Sample Report Output  
```
accuracy is :: 0.9245  
confusion matrix is : [[902  86] [65 947]]  
classification report:
              precision    recall  f1-score   support
           0       0.93      0.91      0.92       988
           1       0.92      0.94      0.93      1012
    accuracy                           0.92      2000
   macro avg       0.92      0.92      0.92      2000
weighted avg       0.92      0.92      0.92      2000
```

---

## 🔍 Key Insights  
- Logistic regression performs well on clean, linearly separable data  
- Height and weight are strong predictors of gender  
- Multiprocessing speeds up EDA and improves workflow efficiency

---

## 🏢 Applications  
This model can be adapted for:

- Educational tools for classification  
- Exploratory data analysis in health and fitness  
- Data science training projects

---

## 📁 Dataset Features Description  
- `Height`: Height in inches  
- `Weight`: Weight in pounds  
- `Gender`: 0 (Female), 1 (Male)

---

## 📂 File Structure  
```
WeightHeight/
├── weight_height_logistic.py
├── weight-height.csv
├── requirements.txt
├── README.md
└── Artifacts/
    └── Weight Height/
        ├── Weight_HeightLogistic_HeatMap.png
        ├── Weight_HeightLogistic_ScatterPlot.png
        ├── Weight_HeightLogistic_PredictionScatter.png
        ├── Weight_HeightLogisticModel.joblib
        └── Weight_HeightLogistic.txt
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
Date: 07/10/2025
