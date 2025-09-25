
# 📊 Advertisement Sales Prediction Case Study

## 🧠 Overview  
This case study focuses on predicting sales based on advertising spending across different media channels (TV, Radio, and Newspaper). The project uses Linear Regression to build a predictive model and provides comprehensive analysis through data visualization and model evaluation.

---

## 🎯 Problem Statement  
Predict sales revenue based on advertising expenditure in different media channels to help businesses optimize their advertising budget allocation.

---

## 📂 Dataset  
- **File**: `Advertising.csv`  
- **Size**: 200 records  
- **Features**:
  - 📺 **TV**: Advertising spending on TV (in thousands)
  - 📻 **Radio**: Advertising spending on Radio (in thousands)
  - 📰 **Newspaper**: Advertising spending on Newspaper (in thousands)
- **Target**: 🛒 **Sales** (in thousands of units)

---

## ⚙️ Features

- 📊 **Data Analysis**: Comprehensive exploratory data analysis with statistical summaries  
- 📈 **Visualization**:
  - 🔥 Correlation heatmap  
  - 🔗 Pairplot for feature relationships  
  - 🎯 Actual vs Predicted scatter plot  
  - 📉 Residual analysis plots  
  - 📌 Feature importance visualization  
- 🤖 **Model**: Linear Regression with StandardScaler preprocessing  
- 🧪 **Evaluation**: MSE, RMSE, and R² metrics  
- 💾 **Artifacts**: Automated model saving with timestamps  

---

## 🧠 Technical Implementation

- 🧮 **Algorithm**: Linear Regression  
- 🧼 **Preprocessing**: StandardScaler for feature normalization  
- 🔄 **Pipeline**: Scikit-learn Pipeline for streamlined workflow  
- 🧪 **Validation**: 80/20 train-test split with random state for reproducibility  

---

## 🚀 Usage

### ✅ Prerequisites  
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ Running the Application  
Use the following command-line arguments to control execution:

#### 🏗️ Train the Model
```bash
python AdvertisementLinearRegressionVisualModel.py --train
```
- Loads and cleans the dataset  
- Performs EDA and visualizations  
- Splits data and trains a Linear Regression model  
- Saves model and logs results  

#### 📈 Predict and Evaluate
```bash
python AdvertisementLinearRegressionVisualModel.py --predict
```
- Loads saved model  
- Predicts sales on test data  
- Calculates MSE, RMSE, and R²  
- Logs coefficients and metrics  
- Saves actual vs predicted scatter plot  

#### 📊 Baseline Evaluation
```bash
python AdvertisementLinearRegressionVisualModel.py --baseline
```
- Same as `--predict`, used for quick evaluation or comparison  
- Useful for validating model performance without retraining  

---

## 📤 Output

The application generates:

- 📊 **Model Performance Metrics**: MSE, RMSE, R² score  
- 🖼️ **Visualizations** (saved in `artifacts/plots/`):
  - `correlation_heatmap.png`: Feature correlation matrix  
  - `pairplot.png`: Pairwise feature relationships  
  - `actual_vs_predicted.png`: Model prediction accuracy  
  - `residual_plot.png`: Residual analysis  
  - `residual_distribution.png`: Residual distribution  
  - `feature_importance.png`: Feature coefficients  
- 💾 **Trained Model**: Saved in `artifacts/models/` with timestamp  

---

## 📈 Model Performance

- ✅ **R² Score**: ~0.90+ (90%+ variance explained)  
- 📉 **Low RMSE**: Indicates good prediction accuracy  
- 📌 **Feature Importance**: TV advertising typically shows highest impact  

---

## 🔍 Key Insights

- 📺 TV Advertising usually has the strongest positive correlation with sales  
- 📻 Radio Advertising shows moderate positive correlation  
- 📰 Newspaper Advertising often has the weakest correlation  
- 📊 The model provides interpretable coefficients for business decision-making  

---

## 📁 File Structure

```
Advertisement_Case_Study/
├── AdvertisementLinearRegressionVisualModel.py
├── Advertising.csv
├── requirements.txt
├── README.md
└── artifacts/
    ├── models/
    │   └── marvellous_advertisement_model_*.pkl
    └── plots/
        ├── correlation_heatmap.png
        ├── pairplot.png
        ├── actual_vs_predicted.png
        ├── residual_plot.png
        ├── residual_distribution.png
        └── feature_importance.png
```

---

## 📦 Dependencies

- `pandas >= 2.1.0`  
- `numpy >= 1.25.0`  
- `matplotlib >= 3.8.0`  
- `seaborn >= 0.12.2`  
- `scikit-learn >= 1.3.0`  
- `joblib >= 1.3.2`  

---

## 👩‍💻 Author

**Sakshi Kedari**  
📅 Date: 25/09/2025  
🔗 GitHub: [sakshikedari3](https://github.com/sakshikedari3)  
📧 Email: sakshikedari426@gmail.com  
💼 LinkedIn: [sakshi-kedari-66001637b](https://www.linkedin.com/in/sakshi-kedari-66001637b)
