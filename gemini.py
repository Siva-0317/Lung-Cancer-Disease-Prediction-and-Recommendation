import streamlit as st
import numpy as np
import joblib
import google.generativeai as genai

# ✅ Load Gemini API Key
GEMINI_API_KEY = "your-api-key"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# ✅ Load trained ML model and scaler
try:
    ml_model = joblib.load("svm_cancer_risk_model.pkl")  # Update if needed
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Please check the file paths.")

# ✅ Feature names (MATCH EXACTLY with model training)
feature_names = [
    "Smoking (packs/month)", "Alcohol (drinks/week)", "Family Cancer History", 
    "Physical Activity (hours/week)", "Processed Food Intake (per week)", "BMI", 
    "Age", "Pollution Exposure (scale 1-10)", "Occupational Hazard", 
    "CA 19-9", "Bilirubin", "HIV Status"
]  

st.title("Lung Cancer Risk Prediction & AI-Based Lifestyle Recommendations")
st.write("Enter patient details to predict the cancer risk level and receive AI-generated lifestyle advice.")

# ✅ Collect user input
user_input = {}
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", min_value=0.0, format="%.2f")
    user_input[feature] = value

# ✅ Convert user input to NumPy array
user_input_values = np.array(list(user_input.values())).reshape(1, -1)

# ✅ Function to Generate AI-Based Lifestyle Recommendations (Dynamic Prompt Engineering)
def get_gemini_recommendation(risk_level, input_data):
    user_data_str = "\n".join(f"{feature}: {value}" for feature, value in input_data.items())

    # 📝 **Different prompt structures based on risk level**
    if risk_level == 0:
        prompt = f"""
        A user has provided the following health data:

        {user_data_str}

        The predicted cancer risk level is LOW (0). This means the user is maintaining a generally healthy lifestyle.
        Provide **brief** lifestyle advice to help them **continue good habits**, and include words of encouragement.
        """
    elif risk_level == 1:
        prompt = f"""
        A user has provided the following health data:

        {user_data_str}

        The predicted cancer risk level is MEDIUM (1). There are **some risk factors** that could be improved.
        Suggest **moderately detailed** lifestyle adjustments that will help lower their risk over time.
        Focus on **practical** and **sustainable** changes.
        """
    else:
        prompt = f"""
        A user has provided the following health data:

        {user_data_str}

        The predicted cancer risk level is HIGH (2). The user has **several risk factors** that may increase their chances of cancer.
        Provide **comprehensive lifestyle recommendations** to significantly reduce cancer risk.
        Emphasize **urgent changes**, potential health risks, and the importance of early screening.
        """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text if response else "No recommendation available."
    except Exception as e:
        return f"Error generating recommendation: {e}"

# ✅ Predict Cancer Risk & Generate Recommendations
if st.button("Predict Cancer Risk & Get Advice"):
    try:
        # Scale input
        user_input_scaled = scaler.transform(user_input_values)

        # Predict Cancer Risk Level
        predicted_risk = int(ml_model.predict(user_input_scaled)[0])
        st.subheader(f"Predicted Cancer Risk Level: {predicted_risk}")

        # Get AI-Generated Lifestyle Recommendation
        recommendation = get_gemini_recommendation(predicted_risk, user_input)

        # Display Results
        st.subheader("AI-Generated Lifestyle Recommendation")
        st.write(recommendation)

    except ValueError as e:
        st.error(f"Prediction Error: {e}")
