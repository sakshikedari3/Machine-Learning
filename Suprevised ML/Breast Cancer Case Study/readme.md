#### Breast Cancer Classification Case Study

#### 

###### Overview



This case study focuses on binary classification of breast cancer tumors as malignant or benign using Random Forest Classifier. The project uses the Wisconsin Breast Cancer dataset and provides comprehensive analysis through data visualization and model evaluation.



###### Problem Statement



Classify breast cancer tumors as malignant (cancerous) or benign (non-cancerous) based on various cell nucleus measurements to assist in early diagnosis and treatment planning.



###### Dataset



* Source: Wisconsin Breast Cancer dataset (CSV format)
* Features: 10 numerical features describing cell nucleus characteristics
* Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses
* Target: Binary classification (2: Benign, 4: Malignant)
* Size: 699 samples
* download : https://github.com/sakshikedari3/Machine-Learning/blob/main/Suprevised%20ML/Breast%20Cancer%20Case%20Study/breast-cancer-wisconsin.csv



###### Features



\- \*\*Data Analysis\*\*: Statistical summaries, datatype inspection, and column overview

\- \*\*Visualization\*\*:

&nbsp; - Correlation heatmap for feature relationships

&nbsp; - Confusion matrix visualization

&nbsp; - Feature importance bar chart

&nbsp; - ROC curve and pairplot

\- **Model**: Random Forest Classifier with 300 estimators

\- Evaluation: Accuracy, Classification Report, Confusion Matrix, AUC Score

\- \*\*Artifacts\*\*: All outputs saved in `Artifacts/Breast\_Cancer` folder



###### Technical Implementation



\- \*\*Algorithm\*\*: Random Forest Classifier

\- \*\*Preprocessing\*\*:

&nbsp; - Label encoding for categorical features

&nbsp; - StandardScaler for feature normalization

\- \*\*Pipeline\*\*: Modular Python functions for each step

\- \*\*Validation\*\*: 80/20 train-test split with fixed random state for reproducibility



#### Usage



###### Prerequisites



Install the required dependencies:



```bash

pip install -r requirements.txt

```



###### Running the Application



To train the model:



```bash

python BreastCancer.py --train

```



###### To test the model:



```bash

python BreastCancer.py --test

```



##### Output



###### The application generates:



\- \*\*Model Performance Metrics\*\*: Accuracy and classification report

\- \*\*Visualizations\*\*: Saved in `Artifacts/Breast\_Cancer/`

&nbsp; - `heatmap.png`: Feature correlation matrix

&nbsp; - `confusion\_matrix.png`: Model prediction accuracy matrix

&nbsp; - `feature\_importance.png`: Random Forest feature importance

&nbsp; - `roc\_curve.png`: ROC curve for classification

&nbsp; - `pair\_plot.png`: Pairwise feature relationships

\- \*\*Trained Model\*\*: Saved as `breast\_cancer\_model.joblib` in `Artifacts/Breast\_Cancer`

\- \*\*Log File\*\*: `Breast\_Cancer\_report.txt` containing all metrics and summaries



###### Model Performance



The Random Forest model typically achieves:



\- Accuracy: 96%+ on test data

\- High Precision and Recall: For both malignant and benign classes

\- Feature Importance: Identifies most discriminative features for classification



###### Key Insights



\- \*\*Feature Selection\*\*: Random Forest highlights key predictors of malignancy

\- \*\*High Accuracy\*\*: Robust performance on real-world medical data

\- \*\*Medical Relevance\*\*: Supports early cancer detection and diagnosis

\- \*\*Interpretability\*\*: Visualizations and metrics aid clinical understanding



###### Dataset Features



The dataset includes 10 core features derived from digitized images of fine needle aspirate (FNA) of breast mass:



\- Clump Thickness

\- Uniformity of Cell Size

\- Uniformity of Cell Shape

\- Marginal Adhesion

\- Single Epithelial Cell Size

\- Bare Nuclei

\- Bland Chromatin

\- Normal Nucleoli

\- Mitoses

\- Cancer Type (Target)



##### File Structure



```

Breast\_Cancer\_Case\_Study/

├── BreastCancer.py

├── requirements.txt

├── README.md

└── Artifacts/

&nbsp;   └── Breast\_Cancer/

&nbsp;       ├── Breast\_Cancer\_report.txt

&nbsp;       ├── breast\_cancer\_model.joblib

&nbsp;       ├── heatmap.png

&nbsp;       ├── confusion\_matrix.png

&nbsp;       ├── feature\_importance.png

&nbsp;       ├── roc\_curve.png

&nbsp;       └── pair\_plot.png

```



##### Dependencies



\- pandas >= 2.1.0  

\- numpy >= 1.25.0  

\- matplotlib >= 3.8.0  

\- seaborn >= 0.12.2  

\- scikit-learn >= 1.3.0  

\- joblib >= 1.3.2  



\## Medical Context



This model can be used as a preliminary screening tool in medical diagnosis:



\- \*\*Benign (2)\*\*: Non-cancerous tumor, typically requires monitoring  

\- \*\*Malignant (4)\*\*: Cancerous tumor, requires immediate medical attention  

\- \*\*Early Detection\*\*: Helps in identifying potential cancer cases early



\## Author



Sakshi Kedari  

Date: 22/09/2025



