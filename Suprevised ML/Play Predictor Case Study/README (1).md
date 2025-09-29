# 🎯 PlayPredictor Case Study

## 📘 Overview  
This case study focuses on predicting whether a game should be played based on weather conditions using a K-Nearest Neighbors (KNN) classifier. The project demonstrates a reproducible machine learning pipeline with modular preprocessing, training, evaluation, and artifact logging.

---

## 🧩 Problem Statement  
Predict whether a game will be played based on weather conditions such as temperature and forecast. This helps simulate decision-making logic for outdoor activities based on environmental factors.

---

## 📊 Dataset  
**File:** PlayPredictor.csv  
**Features:**  
- Weather: `Whether` (Sunny, Overcast, Rainy)  
- Temperature: `Temperature` (Hot, Mild, Cool)  
- Target: `Play` (Yes/No)  
**Size:** 14 records (toy dataset for classification)

---

## ⚙️ Features

### 🔄 Data Preprocessing  
- Categorical feature encoding using manual mapping (`Sunny` → 0, etc.)  
- Feature scaling with `StandardScaler`  
- Removal of irrelevant columns not required for modeling

### 📈 Visualization  
- Correlation heatmap using seaborn  
- Confusion matrix and scatter plot for model performance

### 🤖 Model  
- Algorithm: K-Nearest Neighbors (KNN)  
- Hyperparameter: `n_neighbors=9`  
- Evaluation: Accuracy, Confusion Matrix, Classification Report  
- Artifacts: Model and plots saved automatically

---

## 🧪 Technical Implementation  
- **Preprocessing:** Manual encoding for categorical variables, `StandardScaler` for feature normalization  
- **Pipeline:** Modular Python functions for each step  
- **Validation:** 80/20 train-test split with `random_state=42`  
- **Model Saving:** Using `joblib` with reproducible file paths

---

## 🚀 Usage

### 📦 Prerequisites  
Install required dependencies:
```bash
pip install pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
```

### ▶️ Running the Application  
```bash
python PlayPredictor.py --train   # Train the model  
python PlayPredictor.py --test    # Evaluate the model  
```

### 🧾 Command Line Arguments  
- `--train`: Trains the model and saves artifacts  
- `--test`: Loads the model and evaluates performance  

---

## 📤 Output  
The application generates:

- **Model Performance Metrics:** Accuracy and classification report  
- **Visualizations:** Saved in `Artifacts/PlayPredictor/`  
  - `heatmap.png`: Correlation heatmap  
  - `confusion_matrix.png`: Confusion matrix  
  - `scatter_plot.png`: Scatter plot of predictions  
- **Trained Model:** Saved as `playpredictor_model.joblib`  

---

## 📊 Model Performance  
- Accuracy: ~85–100% (on toy dataset)  
- Precision and Recall: Varies by class balance  
- Feature Importance: Not applicable for KNN, but correlation heatmap provides insights

---

## 🔍 Key Insights  
- Sunny weather with hot temperature often leads to "No Play"  
- Overcast conditions are more likely to result in "Play"  
- Mild and cool temperatures show mixed behavior depending on weather

---

## 🏢 Business Applications  
This model can be adapted for:

- Event Planning: Automate decisions for outdoor activities  
- Sports Scheduling: Predict playability based on weather  
- Educational Tools: Teach ML concepts using interpretable datasets

---

## 📁 Dataset Features Description  
- `Whether`: Weather condition (Sunny, Overcast, Rainy)  
- `Temperature`: Temperature level (Hot, Mild, Cool)  
- `Play`: Target variable (Yes/No)

---

## 📂 File Structure  
```
PlayPredictor/
├── PlayPredictor.py
├── PlayPredictor.csv
├── requirements.txt
├── README.md
└── Artifacts/
    └── PlayPredictor/
        ├── heatmap.png
        ├── confusion_matrix.png
        ├── scatter_plot.png
        └── playpredictor_model.joblib
```

---

## 🧠 Churn Prevention Strategies (Adapted for PlayPredictor)  
While churn isn’t the focus here, similar strategies apply for decision logic:

- Weather-Based Alerts: Notify users when conditions are favorable  
- Activity Recommendations: Suggest alternatives based on forecast  
- Historical Patterns: Learn from past decisions to improve future predictions

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
Date: 29/09/2025

