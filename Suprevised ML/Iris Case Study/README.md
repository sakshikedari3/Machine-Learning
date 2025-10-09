## 🌸 Iris Decision Tree Case Study

### 📖 Overview
This case study focuses on predicting iris flower species using a Decision Tree Classifier. The project analyzes floral measurements to classify iris varieties, demonstrating core machine learning concepts and reproducible workflow design.

---

### ❓ Problem Statement
Predict the species of iris flowers based on sepal and petal dimensions to assist in botanical classification and demonstrate supervised learning techniques.

---

### 📊 Dataset
- **File**: `iris.csv`
- **Features**:
  - `sepal.length`: Sepal length in cm
  - `sepal.width`: Sepal width in cm
  - `petal.length`: Petal length in cm
  - `petal.width`: Petal width in cm
  - `variety`: Iris species (Iris-setosa, Iris-versicolor, Iris-virginica)
- **Size**: 150 records

---

### 🧠 Features
- **Data Analysis**: Descriptive statistics, column types, and sample preview
- **Visualization**:
  - Correlation heatmap
  - Pairplot of features
  - Decision tree plot
- **Model**: Decision Tree Classifier with label encoding
- **Evaluation**: Accuracy score and prediction report
- **Artifacts**: Automated saving of model, plots, and report

---

### ⚙️ Technical Implementation
- **Algorithm**: Decision Tree Classifier
- **Preprocessing**: LabelEncoder for categorical features
- **Validation**: 70/30 train-test split with reproducibility
- **Logging**: Text report and visual artifacts saved to disk

---

### 🧪 Usage

#### 📦 Prerequisites
Install required dependencies:
```bash
pip install pandas seaborn matplotlib scikit-learn joblib
```

#### 🚀 Running the Application
```bash
python iris_decision_tree.py --train     # Train and save the model
python iris_decision_tree.py --predict   # Load model and evaluate accuracy
```

#### 🧾 Command Line Arguments
- `--train`: Train the model and save artifacts
- `--predict`: Load saved model and evaluate performance

---

### 📤 Output
The application generates:

- **Model Performance Metrics**: Accuracy score
- **Visualizations**: Saved in `Artifacts/Iris_DecisionTree/`
  - `Correlation_plot.png`: Feature correlation matrix
  - `Pairplot_plot.png`: Feature pairwise relationships
  - `HeatMap.png`: Feature heatmap
  - `featuer_importance.png`: Decision tree visualization
- **Trained Model**: Saved as `Iris_DecisionTree_model_v1.joblib`
- **Report**: Logged in `Iris_DecisionTree_report.txt`

---

### 📈 Model Performance
- **Accuracy**: Varies based on train-test split
- **Feature Importance**: Visualized via decision tree
- **Insights**:
  - Petal dimensions are highly predictive
  - Sepal features show moderate correlation

---

### 📂 File Structure
```
Iris_DecisionTree/
├── iris_decision_tree.py
├── iris.csv
├── Iris_DecisionTree_report.txt
├── Iris_DecisionTree_model_v1.joblib
├── Correlation_plot.png
├── Pairplot_plot.png
├── HeatMap.png
└── featuer_importance.png
```

---

### 🌿 Botanical Context
This model can be used for:

- Educational purposes in ML and botany
- Automated classification in botanical research
- Demonstrating decision tree interpretability

---

### 🔍 Risk Factors Identified
- **Petal Length & Width**: Strongest predictors of species
- **Sepal Width**: Least correlated with target
- **Variety Encoding**: Enables classification

---

### 📦 Dependencies
- `pandas >= 2.1.0`
- `numpy >= 1.25.0`
- `matplotlib >= 3.8.0`
- `seaborn >= 0.12.2`
- `scikit-learn >= 1.3.0`
- `joblib >= 1.3.2`

---

### 👩‍💻 Author
**Sakshi Kedari**    
Date: 09/10/2025
