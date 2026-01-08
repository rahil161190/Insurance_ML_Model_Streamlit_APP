import pandas as pd
import streamlit as st
import pickle

try:
    with open('Insurance.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'Insurance.pkl' not found. Please ensure it's in the correct directory.")
    st.stop()

st.title('Insurance Premium Prediction App')
st.write('Enter the client details to predict the insurance premium.')


age = st.number_input('Age', min_value=18, max_value=66)

st.subheader('Height')
height_feet = st.number_input('Feet', min_value=4, max_value=7, key='height_ft')
height_inches = st.number_input('Inches', min_value=0, max_value=11, key='height_in')

height_cm = (height_feet * 30.48) + (height_inches * 2.54)

st.info(f'Converted Height: {height_cm:.2f} cm')

weight = st.number_input('Weight (kg)', min_value=51, max_value=132)
number_of_major_surgeries = st.slider('Number of Major Surgeries', 0, 3, 0)

diabetes = st.radio('Diabetes', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
blood_pressure_problems = st.radio('Blood Pressure Problems', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
any_transplants = st.radio('Any Transplants', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
any_chronic_diseases = st.radio('Any Chronic Diseases', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
known_allergies = st.radio('Known Allergies', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
history_of_cancer_in_family = st.radio('History of Cancer in Family', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

def model_predict(
    age_val, diabetes_val, blood_pressure_problems_val, any_transplants_val,
    any_chronic_diseases_val, height_val, weight_val, known_allergies_val,
    history_of_cancer_in_family_val, number_of_major_surgeries_val
):

    input_data = pd.DataFrame([[
        age_val, diabetes_val, blood_pressure_problems_val, any_transplants_val,
        any_chronic_diseases_val, height_val, weight_val, known_allergies_val,
        history_of_cancer_in_family_val, number_of_major_surgeries_val
    ]],
    columns=[
        'Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
        'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies',
        'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries'
    ])

    return model.predict(input_data)

if st.button("Predict Insurance Premium"):
    predicted_premium = model_predict(
        age, diabetes, blood_pressure_problems, any_transplants,
        any_chronic_diseases, height_cm, weight, known_allergies,
        history_of_cancer_in_family, number_of_major_surgeries
    )[0]
    st.success(f"Predicted Insurance Premium: ₹{predicted_premium:.2f}")


st.markdown("--- App Information ---")
st.info("This app uses a trained machine learning model to predict insurance premiums based on provided health details.")


