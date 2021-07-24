import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Title
image = Image.open('featured.png')

st.image(image, width=600)
st.title('Heart Strokes Predictions App')
st.markdown("""
This app predicts the heart strokes of patients by adding up there current health conditions and other basic informations of patients.
""")

st.sidebar.header('User Input Features')

# gender = data.groupby('gender')
# age = data.groupby('age')
# hypertension = data.groupby('hypertension')
# heart_disease = data.groupby('heart_disease')
# ever_married = data.groupby('ever_married')
# work_type = data.groupby('work_type')
# Residence_type = data.groupby('Residence_type')
# avg_glucose_level = data.groupby('avg_glucose_level')
# bmi = data.groupby('bmi')
# smoking_status = data.groupby('smoking_status')

# Sidebar
# Collects user input features into dataframe
def user_input_features():
    gender = st.sidebar.slider('Gender', 0, 2, 1)
    age = st.sidebar.slider('Age', 0, 104, 88)
    hypertension = st.sidebar.slider('Hypertension', 0, 2, 0)
    heart_disease = st.sidebar.slider('Heart Diseases', 0, 1, 1)
    ever_married = st.sidebar.slider('Married', 0, 1, 1)
    work_type = st.sidebar.slider('Work Type', 0, 3, 2)
    Residence_type = st.sidebar.slider('Residence Type', 0, 1, 1)
    avg_glucose_level = st.sidebar.slider('Avg Glucose level', 0, 4000, 3850)
    bmi = st.sidebar.slider('BMI', 0, 500, 239)
    smoking_status = st.sidebar.slider('Smoking Status', 0, 3, 1)
    data = {'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status}
    features = pd.DataFrame(data, index= [0])
    return features

input_df = user_input_features()

data_raw = pd.read_csv('cleaned_data.csv')
new_data = data_raw.drop(['stroke'], axis=1)
df = pd.concat([input_df, new_data])
df_1 = df.iloc[:,:-1]
print(df_1)

# prediction = model.predict([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, smoking_status]])

# Displays the user input features
st.subheader('User Input features')
st.markdown("""
Gender: **Male, Female, Other**

Hypertension: **Yes, NO**

Heart Diseases: **Yes, NO**

Married: **Yes, NO**

Work Type: **Private, Self-employed, Govt_job, children, Never_worked**

Residence Type: **Urban, Rural**

Smoking Status: **Formerly-Smoked, Never-Smoked, Smokes, Unknown**

""")
df_2 = df_1[:1]  # Selects only the first row (the user input data)

st.write(df_2)

# reading model file
pickle_in = open("model.pkl", "rb")
load_model = pickle.load(pickle_in)

# Apply model to make predictions
prediction = load_model.predict(df_2)
prediction_proba = load_model.predict_proba(df_2)

# st.subheader('Prediction')
# stroke = np.array(['No stroke', 'Stroke'])
# st.write(stroke[prediction])


st.subheader('Prediction Probability')
st.write(prediction_proba)