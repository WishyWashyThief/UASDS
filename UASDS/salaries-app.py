import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

st.write("""
# Salary Prediction App

This app predicts your **Annual Salary in Dollars($)**!

Data obtained from the [salary prediction for beginer](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer).
""")

st.sidebar.header('User Input Features')

def user_input_features():
    edu = st.sidebar.selectbox('Education Level',("Bachelor's","Master's","PhD"))
    # sex = st.sidebar.selectbox('Sex',('male','female'))
    age = st.sidebar.slider('Age', 22, 60)
    yoe = st.sidebar.slider('Years Of Experience', 0, 40)
    # body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
    data = {'Education Level': edu,
            'Age': age,
            'Years of Experience': yoe
            #'body_mass_g': body_mass_g,
            #'sex': sex
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

salaries_raw = pd.read_csv('salary data.csv').dropna()
salaries = salaries_raw.drop(columns = ["Gender", "Job Title", "Salary"])
df = pd.concat([input_df, salaries], axis = 0)

encode = ['Education Level']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df,dummy], axis = 1)
    del df[col]

df = df[:1]


# Displays the user input features
st.subheader('User Input features')
st.write(df)

# Reads in saved classification model
load_lr = pickle.load(open('salaries_lr.pkl', 'rb'))

# Apply model to make predictions
prediction = load_lr.predict(df)

st.subheader('Prediction')
st.write(prediction)

 
# # Combines user input features with entire penguins dataset
# # This will be useful for the encoding phase
# penguins_raw = pd.read_csv('penguins_cleaned.csv')
# penguins = penguins_raw.drop(columns=['species'])
# df = pd.concat([input_df,penguins],axis=0)

# # Encoding of ordinal features
# # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:] # Selects only the first row (the user input data)

# # Displays the user input features
# st.subheader('User Input features')

# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
#     st.write(df)

# # Reads in saved classification model
# load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# # Apply model to make predictions
# prediction = load_clf.predict(df)
# prediction_proba = load_clf.predict_proba(df)

# st.subheader('Prediction')
# penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
# st.write(penguins_species[prediction])

# st.subheader('Prediction Probability')
# st.write(prediction_proba)
