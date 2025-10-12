
# 🧬 Breast Cancer Classification with SVM

## 📘 Overview  
This project predicts whether a tumor is malignant or benign using a Support Vector Machine (SVM) model trained on the Breast Cancer Wisconsin dataset. It demonstrates a reproducible classification workflow with modular preprocessing, training, evaluation, and artifact logging.

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
**Size:** 698 records

---

## ⚙️ Features

### 🔄 Data Preprocessing  
- Label encoding for categorical features  
- Feature scaling with `StandardScaler`  
- Dropped non-informative columns: `CodeNumber`, `CancerType`  
- Train-test split with reproducibility

### 📈 Visualization  
- Heatmap of feature correlations  
- Confusion matrix (visual)  
- Feature importance bar chart (if supported)

### 🤖 Model  
- Algorithm: Support Vector Machine (SVC with linear kernel)  
- Evaluation: Accuracy, Confusion Matrix, Classification Report, AUC Score  
- Artifacts: Model and plots saved automatically

---

## 🧪 Technical Implementation  
- Modular Python functions for each step  
- Model saving and loading with `joblib`  
- Command-line interface using `--train` and `--test` flags  
- Visual artifacts saved to `Artifacts/Breast_CancerSVM/`

---

## 🚀 Usage

### 📦 Prerequisites  
Install required dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ Running the Application  
```bash
python breast_cancer_svm.py --train     # Train the model  
python breast_cancer_svm.py --test      # Evaluate the model  
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
- **Visualizations:** Saved in `Artifacts/Breast_CancerSVM/`  
  - `heatmap.png`  
  - `confusion_matrix.png`  
  - `feature_importance.png` (if supported)  
- **Trained Model:**  
  - `breast_cancer_SVMmodel.joblib`  
- **Logs:** Written to `Breast_CancerSVM_report.txt`

---

## 📊 Sample Report Output  
```
Training accuracy is : 0.9606  
Testing accuracy is : 0.9786  
Confusion matrix:
[[86  0]
 [ 3 51]]  
Classification report:
              precision    recall  f1-score   support
           2       0.97      1.00      0.98        86
           4       1.00      0.94      0.97        54
AUC Score: 0.9831
```

---

## 🔍 Key Insights  
- SVM with a linear kernel performs exceptionally well on this medical dataset  
- Feature scaling is essential for SVM performance  
- AUC score and confusion matrix provide confidence in model reliability

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
BreastCancerSVM/
├── breast_cancer_svm.py
├── breast-cancer-wisconsin.csv
├── requirements.txt
├── README.md
└── Artifacts/
    └── Breast_CancerSVM/
        ├── heatmap.png
        ├── confusion_matrix.png
        ├── feature_importance.png
        ├── breast_cancer_SVMmodel.joblib
        └── Breast_CancerSVM_report.txt
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
Date: 13/10/2025
