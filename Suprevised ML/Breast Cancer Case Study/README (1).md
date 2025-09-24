# 🩺 Breast Cancer Classification Case Study

## 📌 Overview
This case study focuses on binary classification of breast cancer tumors as malignant or benign using a Random Forest Classifier. It uses the Wisconsin Breast Cancer dataset and provides comprehensive analysis through data visualization and model evaluation.

## 🎯 Problem Statement
Classify breast cancer tumors as malignant (cancerous) or benign (non-cancerous) based on various cell nucleus measurements to assist in early diagnosis and treatment planning.

## 📊 Dataset
- **Source**: [Wisconsin Breast Cancer dataset](https://github.com/sakshikedari3/Machine-Learning/blob/main/Suprevised%20ML/Breast%20Cancer%20Case%20Study/breast-cancer-wisconsin.csv)
- **Samples**: 699
- **Target**: Binary classification (2: Benign, 4: Malignant)
- **Features**:
  - Clump Thickness
  - Uniformity of Cell Size
  - Uniformity of Cell Shape
  - Marginal Adhesion
  - Single Epithelial Cell Size
  - Bare Nuclei
  - Bland Chromatin
  - Normal Nucleoli
  - Mitoses

## ⚙️ Features

### 🔍 Data Analysis
- Statistical summaries
- Datatype inspection
- Column overview

### 📈 Visualization
- Correlation heatmap
- Confusion matrix
- Feature importance bar chart
- ROC curve
- Pairplot

### 🤖 Model
- Random Forest Classifier with 300 estimators

### 🧪 Evaluation
- Accuracy
- Classification Report
- Confusion Matrix
- AUC Score

### 📦 Artifacts
All outputs saved in `Artifacts/Breast_Cancer/`

## 🧠 Technical Implementation
- **Algorithm**: Random Forest Classifier
- **Preprocessing**:
  - Label encoding for categorical features
  - StandardScaler for feature normalization
- **Pipeline**: Modular Python functions
- **Validation**: 80/20 train-test split with fixed random state

## 🚀 Usage

### ✅ Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ Running the Application
To train the model:
```bash
python BreastCancer.py --train
```

To test the model:
```bash
python BreastCancer.py --test
```

## 📤 Output

### 📊 Model Performance Metrics
- Accuracy
- Classification report

### 📈 Visualizations (saved in `Artifacts/Breast_Cancer/`)
- `heatmap.png`: Feature correlation matrix
- `confusion_matrix.png`: Prediction accuracy matrix
- `feature_importance.png`: Feature importance chart
- `roc_curve.png`: ROC curve
- `pair_plot.png`: Pairwise feature relationships

### 💾 Trained Model
- `breast_cancer_model.joblib`

### 📝 Log File
- `Breast_Cancer_report.txt`

## 📈 Model Performance
- Accuracy: 96%+
- High precision and recall for both classes
- Highlights most discriminative features

## 🔍 Key Insights
- Random Forest identifies key predictors of malignancy
- High accuracy on real-world medical data
- Supports early cancer detection
- Visualizations aid clinical understanding

## 🧬 Dataset Features
Derived from digitized images of fine needle aspirate (FNA) of breast mass:
- Clump Thickness
- Uniformity of Cell Size
- Uniformity of Cell Shape
- Marginal Adhesion
- Single Epithelial Cell Size
- Bare Nuclei
- Bland Chromatin
- Normal Nucleoli
- Mitoses
- Cancer Type (Target)

## 📁 File Structure
```
Breast_Cancer_Case_Study/
├── BreastCancer.py
├── requirements.txt
├── README.md
└── Artifacts/
    └── Breast_Cancer/
        ├── Breast_Cancer_report.txt
        ├── breast_cancer_model.joblib
        ├── heatmap.png
        ├── confusion_matrix.png
        ├── feature_importance.png
        ├── roc_curve.png
        └── pair_plot.png
```

## 🏥 Medical Context
This model can assist in preliminary screening:
- **Benign (2)**: Non-cancerous, typically monitored
- **Malignant (4)**: Cancerous, requires immediate attention
- **Early Detection**: Enables timely intervention

## 📦 Dependencies
- `pandas >= 2.1.0`
- `numpy >= 1.25.0`
- `matplotlib >= 3.8.0`
- `seaborn >= 0.12.2`
- `scikit-learn >= 1.3.0`
- `joblib >= 1.3.2`

## 👩‍💻 Author
**Sakshi Kedari**  
Date: 22/09/2025
