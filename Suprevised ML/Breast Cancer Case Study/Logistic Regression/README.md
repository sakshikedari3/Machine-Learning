
# 🧬 Breast Cancer Classification with Logistic Regression

## 📘 Overview  
This project predicts whether a tumor is malignant or benign using logistic regression on the Breast Cancer Wisconsin dataset. It demonstrates a reproducible classification workflow with modular preprocessing, training, evaluation, and artifact logging.

---

## 🧩 Problem Statement  
Classify tumors based on cell-level features to support early diagnosis and reduce false positives in cancer screening.

---

## 📊 Dataset  
**File:** `breast-cancer-wisconsin.csv`  
**Features:**  
- ClumpThickness  
- UniformityCellSize  
- UniformityCellShape  
- MarginalAdhesion  
- SingleEpithelialCellSize  
- BareNuclei  
- BlandChromatin  
- NormalNucleoli  
- Mitoses  
- Target: `CancerType` (2 = benign, 4 = malignant)  
**Size:** 699 records

---

## ⚙️ Features

### 🔄 Data Preprocessing  
- Replaced missing values in `BareNuclei`  
- Converted `BareNuclei` to numeric  
- Label encoding for categorical features  
- Feature scaling with `StandardScaler`  
- Train-test split with reproducibility

### 📈 Visualization  
- Heatmap of feature correlations  
- Pairplot of feature relationships  
- Confusion matrix (visual)  
- Feature importance bar chart  
- ROC curve

### 🤖 Model  
- Algorithm: Logistic Regression  
- Evaluation: Accuracy, Confusion Matrix, Classification Report, AUC Score  
- Artifacts: Model and plots saved automatically

---

## 🧪 Technical Implementation  
- Modular Python functions for each step  
- Model saving and loading with `joblib`  
- Command-line interface using `--train` and `--test` flags  
- Visual artifacts saved to `Artifacts/Breast_Cancer/`

---

## 🚀 Usage

### 📦 Prerequisites  
Install required dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ Running the Application  
```bash
python breast_cancer_logistic.py --train     # Train the model  
python breast_cancer_logistic.py --test      # Evaluate the model  
```

### 🧾 Command Line Arguments  
- `--train`: Trains the model and saves artifacts  
- `--test`: Loads the model and evaluates performance

---

## 📤 Output  
The application generates:

- **Model Performance Metrics:**  
  - Accuracy  
  - Confusion Matrix  
  - Classification Report  
  - AUC Score  
- **Visualizations:** Saved in `Artifacts/Breast_Cancer/`  
  - `logisticHeatmap.png`  
  - `logisticConfusion_matrix.png`  
  - `logisticPair_plot.png`  
  - `logisticFeature_importance.png`  
  - `logisticRoc_curve.png`  
- **Trained Model:**  
  - `breast_cancer_model_Logistic.joblib`  
- **Logs:** Written to `Breast_Cancer_report_logistic.txt`

---

## 📊 Sample Report Output  
```
Testing accuracy is : 0.9714  
Confusion matrix is :
[[86  3]
 [ 1 50]]  
Classification report:
              precision    recall  f1-score   support
           2       0.99      0.97      0.98        89
           4       0.94      0.98      0.96        51
AUC Score: 0.9733
```

---

## 🔍 Key Insights  
- Logistic regression performs well on clean, linearly separable medical data  
- `BareNuclei`, `UniformityCellSize`, and `ClumpThickness` are strong predictors  
- ROC curve and AUC score provide confidence in model reliability

---

## 🏥 Applications  
This model can be adapted for:

- Clinical decision support  
- Cancer screening automation  
- Educational tools for medical ML  
- Data science training projects

---

## 📁 Dataset Features Description  
- `ClumpThickness`: Thickness of cell clusters  
- `BareNuclei`: Count of bare nuclei  
- `CancerType`: 2 = benign, 4 = malignant

---

## 📂 File Structure  
```
BreastCancer/
├── breast_cancer_logistic.py
├── breast-cancer-wisconsin.csv
├── requirements.txt
├── README.md
└── Artifacts/
    └── Breast_Cancer/
        ├── logisticHeatmap.png
        ├── logisticConfusion_matrix.png
        ├── logisticPair_plot.png
        ├── logisticFeature_importance.png
        ├── logisticRoc_curve.png
        ├── breast_cancer_model_Logistic.joblib
        └── Breast_Cancer_report_logistic.txt
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
Date: 10/10/2025

