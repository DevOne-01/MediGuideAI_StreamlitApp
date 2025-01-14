import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

def load_and_preprocess_data():
    # Replace this with your dataset path
    file_path = "data/medical data.csv"  # Ensure this file is in the same directory
    data = pd.read_csv(file_path)

    # Convert DateOfBirth to Age
    data['DateOfBirth'] = pd.to_datetime(data['DateOfBirth'], format='%d-%m-%Y', errors='coerce')
    data['Age'] = (pd.Timestamp('today') - data['DateOfBirth']).dt.days // 365

    # Drop irrelevant columns
    data_cleaned = data.drop(columns=['Name', 'DateOfBirth'])

    # Handle missing values
    data_cleaned['Age'] = data_cleaned['Age'].fillna(data_cleaned['Age'].median())

    # Encode Gender
    gender_encoder = OneHotEncoder(drop='first', sparse_output=False)
    gender_encoded = gender_encoder.fit_transform(data_cleaned[['Gender']])
    gender_encoded_df = pd.DataFrame(gender_encoded, columns=gender_encoder.get_feature_names_out(['Gender']))

    # Encode Symptoms
    symptom_list = data_cleaned['Symptoms'].str.split(', ', expand=True).stack().unique()
    symptom_df = pd.DataFrame(0, index=data_cleaned.index, columns=symptom_list)
    for index, symptoms in data_cleaned['Symptoms'].fillna('').items():
        for symptom in symptoms.split(', '):
            if symptom in symptom_list:
                symptom_df.loc[index, symptom] = 1

    # Combine all features
    data_features = pd.concat([data_cleaned[['Age']], gender_encoded_df, symptom_df], axis=1)

    # Encode Causes and Disease
    cause_encoder = LabelEncoder()
    data_cleaned['Cause_Label'] = cause_encoder.fit_transform(data_cleaned['Causes'])

    disease_encoder = LabelEncoder()
    data_cleaned['Disease_Label'] = disease_encoder.fit_transform(data_cleaned['Disease'])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_features, 
        data_cleaned[['Cause_Label', 'Disease_Label']], 
        test_size=0.2, 
        random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, gender_encoder, symptom_list, cause_encoder, disease_encoder, data_cleaned

# Train models
def train_models(X_train_scaled, y_train):
    cause_model = GradientBoostingClassifier(random_state=42)
    disease_model = GradientBoostingClassifier(random_state=42)

    cause_model.fit(X_train_scaled, y_train['Cause_Label'])
    disease_model.fit(X_train_scaled, y_train['Disease_Label'])

    return cause_model, disease_model

# Preprocess input
def preprocess_input(gender, age, symptoms, scaler, gender_encoder, symptom_list):
    # Encode gender
    gender_encoded = gender_encoder.transform([[gender]])
    gender_encoded_df = pd.DataFrame(gender_encoded, columns=gender_encoder.get_feature_names_out(['Gender']))

    # Encode symptoms
    symptoms_binary = pd.DataFrame(0, index=[0], columns=symptom_list)
    for symptom in symptoms.split(', '):
        if symptom in symptom_list:
            symptoms_binary.loc[0, symptom] = 1

    # Combine features
    input_data = pd.concat([pd.DataFrame({'Age': [age]}), gender_encoded_df, symptoms_binary], axis=1)
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

# Prediction pipeline
def predict(gender, age, symptoms, cause_model, disease_model, scaler, gender_encoder, symptom_list, cause_encoder, disease_encoder, data_cleaned):
    input_data_scaled = preprocess_input(gender, age, symptoms, scaler, gender_encoder, symptom_list)

    # Predict cause and disease
    cause_label = cause_model.predict(input_data_scaled)[0]
    disease_label = disease_model.predict(input_data_scaled)[0]

    # Decode labels
    predicted_cause = cause_encoder.inverse_transform([cause_label])[0]
    predicted_disease = disease_encoder.inverse_transform([disease_label])[0]

    # Recommend medicine
    medicine = data_cleaned[data_cleaned['Disease'] == predicted_disease]['Medicine'].iloc[0]

    return predicted_cause, predicted_disease, medicine

# Streamlit App UI
st.title("Health Prediction and Medicine Recommendation")
st.write("Enter patient details below to predict the cause, disease, and get medicine recommendations.")

# Load and preprocess data
X_train_scaled, X_test_scaled, y_train, y_test, scaler, gender_encoder, symptom_list, cause_encoder, disease_encoder, data_cleaned = load_and_preprocess_data()

# Train models
cause_model, disease_model = train_models(X_train_scaled, y_train)

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
symptoms = st.text_input("Symptoms", "Fever, Cough")

# Predict and display results
if st.button("Predict"):
    try:
        predicted_cause, predicted_disease, recommended_medicine = predict(
            gender, age, symptoms, cause_model, disease_model, scaler, gender_encoder, symptom_list, cause_encoder, disease_encoder, data_cleaned
        )

        st.subheader("Prediction Results")
        st.write(f"**Predicted Cause:** {predicted_cause}")
        st.write(f"**Predicted Disease:** {predicted_disease}")
        st.write(f"**Recommended Medicine:** {recommended_medicine}")
    except Exception as e:
        st.error(f"An error occurred: {e}")