import streamlit as st
import numpy as np
import joblib
import google.generativeai as genai
import pandas as pd
import os

# ✅ Load Gemini API Key
genai.configure(api_key="your-api-key")  # Replace with actual API key
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# ✅ Load trained ML model and scaler
try:
    ml_model = joblib.load("svm_cancer_risk_model.pkl")  # Update if needed
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Please check the file paths.")

# ✅ Feature names
feature_names = [
    "Smoking (packs/month)", "Alcohol (drinks/week)", "Family Cancer History", 
    "Physical Activity (hours/week)", "Processed Food Intake (per week)", "BMI", 
    "Age", "Pollution Exposure (scale 1-10)", "Occupational Hazard", 
    "CA 19-9", "Bilirubin", "HIV Status"
]  

# ✅ User Authentication (Simple username-based tracking)
username = st.text_input("Enter your username:")

# ✅ Define History Storage File
HISTORY_FILE = "user_history.csv"
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["Username", "Inputs", "Risk Level", "Recommendation"]).to_csv(HISTORY_FILE, index=False)

st.title("Lung Cancer Risk Prediction & AI-Based Lifestyle Recommendations")
st.write("Enter patient details to predict cancer risk and receive AI-generated advice.")

# ✅ Collect user input
user_input = {}
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", min_value=0.0, format="%.2f")
    user_input[feature] = value

# ✅ Convert user input to NumPy array
user_input_values = np.array(list(user_input.values())).reshape(1, -1)

def get_gemini_recommendation(risk_level, input_data, user_history):
    user_data_str = "\n".join(f"{feature}: {value}" for feature, value in input_data.items())
    history_str = "\n".join(f"{row['Inputs']} -> Risk Level: {row['Risk Level']}" for _, row in user_history.iterrows())
    
    prompt = f"""
    A user has provided the following health data:
    {user_data_str}
    
    User history of past predictions:
    {history_str}
    
    The predicted cancer risk level is {risk_level}. Based on both the current input and past trends, provide personalized lifestyle recommendations.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text if response else "No recommendation available."
    except Exception as e:
        return f"Error generating recommendation: {e}"

# ✅ Predict & Store History
if st.button("Predict Cancer Risk & Get Advice"):
    if username.strip() == "":
        st.error("Please enter a username to track history.")
    else:
        try:
            user_input_scaled = scaler.transform(user_input_values)
            predicted_risk = int(ml_model.predict(user_input_scaled)[0])
            
            # Load user history
            history_df = pd.read_csv(HISTORY_FILE)
            user_history = history_df[history_df["Username"] == username]
            
            recommendation = get_gemini_recommendation(predicted_risk, user_input, user_history)
            
            st.subheader(f"Predicted Cancer Risk Level: {predicted_risk}")
            st.subheader("AI-Generated Lifestyle Recommendation")
            st.write(recommendation)
            
            # ✅ Save history to CSV
            new_entry = pd.DataFrame({
                "Username": [username],
                "Inputs": [str(user_input)],
                "Risk Level": [predicted_risk],
                "Recommendation": [recommendation]
            })
            history_df = pd.concat([history_df, new_entry], ignore_index=True)
            history_df.to_csv(HISTORY_FILE, index=False)
        
        except ValueError as e:
            st.error(f"Prediction Error: {e}")

# ✅ Display User History
if username.strip() != "":
    st.subheader("Your Past Predictions")
    history_df = pd.read_csv(HISTORY_FILE)
    user_history = history_df[history_df["Username"] == username]
    st.dataframe(user_history)