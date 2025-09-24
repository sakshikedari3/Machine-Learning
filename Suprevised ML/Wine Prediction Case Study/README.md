
# 🍷 Wine Quality Classification Case Study

## 📌 Overview
This case study focuses on wine quality classification using the K-Nearest Neighbors (KNN) algorithm. The project analyzes various chemical properties of wine to predict its quality class, providing insights into wine production and quality assessment.

## 🎯 Problem Statement
Classify wine quality based on various chemical properties and measurements to assist in wine production quality control and help consumers make informed purchasing decisions.

## 📊 Dataset
- **File**: `WinePredictor.csv`
- **Samples**: 178
- **Target**: `Class` (1, 2, 3 — Wine quality classes)
- **Features**:
  - Alcohol
  - Malic acid
  - Ash
  - Alkalinity of ash
  - Magnesium
  - Total phenols
  - Flavanoids
  - Nonflavanoid phenols
  - Proanthocyanins
  - Color intensity
  - Hue
  - OD280/OD315 of diluted wines
  - Proline

## ⚙️ Features

### 🔍 Data Preprocessing
- Missing value handling (`dropna`)
- Feature scaling with `StandardScaler`

### 🔧 Hyperparameter Tuning
- K-value optimization (1–19)
- Accuracy vs K plot for optimal parameter selection

### 📈 Visualization
- Accuracy vs K value plot
- Correlation heatmaps
- Pairplots

### 🤖 Model
- K-Nearest Neighbors Classifier with optimal K

### 🧪 Evaluation
- Accuracy
- Confusion Matrix
- Classification Report

### 📦 Artifacts
- Automated model saving with timestamps
- Visualizations saved in `Artifacts/Wine_Predictor/`

## 🧠 Technical Implementation
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Preprocessing**: `StandardScaler` for feature normalization
- **Hyperparameter Tuning**: Grid search for optimal K value (1–19)
- **Validation**: 70/30 train-test split
- **Distance Metric**: Euclidean (default)

## 🚀 Usage

### ✅ Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ Running the Application
```bash
python wine_predictor.py --train
python wine_predictor.py --predict
```

### 🧾 Command Line Arguments
- `--train`: Train and save the model
- `--predict`: Load model and evaluate accuracy

## 📤 Output
- **Model Performance Metrics**: Accuracy, confusion matrix, classification report
- **Visualizations**:
  - `Correlation_plot.png`
  - `Pairplot_plot.png`
  - `HeatMap.png`
  - `K_value_vs_Accuracy.png`
- **Trained Model**: Saved in `Artifacts/Wine_Predictor/`

## 📊 Model Performance
- **Accuracy**: 90–95% on test data with optimal K
- **Optimal K**: Typically between 3–7 neighbors
- **Precision & Recall**: Strong across all classes

## 🔍 Key Insights
- **Alcohol Content**: Higher alcohol often correlates with better quality
- **Phenolic Compounds**: Flavanoids and total phenols are key indicators
- **Color Properties**: Hue and intensity affect classification
- **KNN Effectiveness**: Performs well on small, well-separated datasets

## 🍷 Wine Quality Classes
- **Class 1**: Lower quality wines
- **Class 2**: Medium quality wines
- **Class 3**: Higher quality wines

## 🧪 Feature Ranges
| Feature                  | Range         |
|--------------------------|---------------|
| Alcohol                  | 11.03–14.83   |
| Malic acid               | 0.74–5.80     |
| Ash                      | 1.36–3.23     |
| Alkalinity of ash        | 10.6–30.0     |
| Magnesium                | 70–162        |
| Total phenols            | 0.98–3.88     |
| Flavanoids               | 0.34–5.08     |
| Nonflavanoid phenols     | 0.13–0.66     |
| Proanthocyanins          | 0.41–3.58     |
| Color intensity          | 1.28–13.0     |
| Hue                      | 0.48–1.71     |
| OD280/OD315              | 1.27–4.00     |
| Proline                  | 278–1680      |

## ✅ KNN Algorithm Benefits
- Non-parametric
- Simple and interpretable
- Effective on small datasets
- Robust to noise

## 🔧 Hyperparameter Tuning Process
- Test K values from 1 to 19
- Use train-test split for evaluation
- Visualize accuracy across K values
- Select K with highest accuracy
- Train final model with optimal K

## 📁 File Structure
```
Wine_Predictor/
├── WinePredictor.csv
├── wine_predictor.py
├── requirements.txt
├── README.md
└── Artifacts/
    └── Wine_Predictor/
        ├── models/
        │   └── WinePredictor_KNN_model_v1.joblib
        └── plots/
            ├── Correlation_plot.png
            ├── Pairplot_plot.png
            ├── HeatMap.png
            └── K_value_vs_Accuracy.png
```

## 🏭 Wine Industry Applications
- Quality Control
- Production Optimization
- Consumer Guidance
- Research
- Competition Judging

## 📦 Dependencies
- `pandas >= 2.1.0`
- `numpy >= 1.25.0`
- `matplotlib >= 3.8.0`
- `seaborn >= 0.12.2`
- `scikit-learn >= 1.3.0`
- `joblib >= 1.3.2`

## 👩‍💻 Author
**Sakshi Kedari**  
