# Heart Strokes Predictions Overview:

- Created a model to predict heart strokes of patients with help of patients health report.
- Dataset used from kaggle platform.
- Using Machine Learning algorithms to predict the strokes of patients based on there reports.
- Random Forest and Decision Tree algorithms had used to reach the best outcomes.
- Created a pickle file to reuse this model in future.

# Code and Libraries used

Python Version - 3.7

IDE - Jupyter Notebook

Packages - Pandas, Numpy, Matplotlib, Plotly, Seaborn, Pickle

# Dataset
- gender	
- age	
- hypertension	
- heart_disease	
- ever_married	
- work_type	
- Residence_type	
- avg_glucose_level	
- bmi	
- smoking_status	
- stroke

# Data Cleaning

- Finded missing values in one column
- Filled those values using mean of it
- Lot of strings in the data, So converted those into numeric data by using label encoder
- Divided stokes occuring people from without stroke people

# Exploratory Data Analysis

Performed some insights on the data after being cleaned.
![download (7)](https://user-images.githubusercontent.com/40689141/117150152-840cc200-add5-11eb-914d-9bb763720f84.png)
![download (8)](https://user-images.githubusercontent.com/40689141/117150178-8a9b3980-add5-11eb-8f18-8953992a2e4d.png)

# Model Building

First, Started splitting data into train and test with the testing size of 20%.

First Model:
         And started, building model using Decision Tree Classifier
         Got an accuracy of 91% which i felt its good

Second Model:
         In this, used Random Forest Classifier with 100 estimators
         Now, i wonder this gave me 95% with only 100 estimators, Surely this would be the great model check it!!
         
# Model Performances

Accuracy Scores
- Random Forest : 95 %
- Decision Tree : 90 %


Note: Don't miss this amazing model, definetly u miss something without checking it out.

Keep in Touch !! https://www.linkedin.com/in/akshith-kumar-469857135/
