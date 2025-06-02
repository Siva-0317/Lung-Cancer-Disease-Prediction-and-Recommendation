🫁 AI-Powered Lung Cancer Risk Prediction & Lifestyle Recommendation System
Top 30 Finalist – Electrothon, Chennai Institute of Technology (2025)

An AI-powered tool for early lung cancer risk detection with personalized lifestyle suggestions using RAG (Retrieval-Augmented Generation).

🔍 Overview
This project combines Machine Learning and Generative AI to predict the risk level of lung cancer based on lifestyle and medical metrics, and provides personalized lifestyle recommendations using Gemini API (Google AI). It aims to assist doctors, caregivers, and patients with proactive cancer prevention strategies.

🎯 Key Features
🧠 Predicts lung cancer risk using a trained SVM model

📊 Input: Lifestyle + biomarker data (e.g., smoking habits, CA 19-9 levels, BMI, HIV status)

💬 Generates Gemini-based personalized lifestyle recommendations

📚 Maintains user-specific prediction history with insights over time

🔐 Lightweight authentication using username-based tracking

🔬 ML Model Details
📌 Model: Support Vector Machine (SVM)
Type: Binary classification (Risk Level: 0 = Low, 1 = High)

Trained on: Synthetic + enriched healthcare dataset

Preprocessing: StandardScaler for normalization

Input Features:

Smoking, Alcohol, BMI, Age, HIV, CA19-9, Bilirubin, etc.

🤖 RAG with Gemini (Google AI)
Model: Gemini 1.5 Pro (via Google Generative AI SDK)

Purpose:

Analyze risk level

Generate health advice using both current input and user history

Output: Tailored, human-like recommendations

🧪 How It Works

[User Inputs] → [Scaler] → [SVM Model] → [Risk Level]
        ↓                                  ↓
   Gemini Prompt ←── [Input Data + User History]
        ↓
[Lifestyle Recommendation]

🛠 Tech Stack
Component	     Tech Used
ML Model	      Scikit-learn (SVM)
Frontend	      Streamlit
AI API	        Google Gemini API (RAG)
Data Storage	  CSV-based user history logging
Language	      Python

🏅 Hackathon Achievement
Selected as a Top 30 Finalist in Electrothon 2025 held at Chennai Institute of Technology, among 200+ competing teams nationwide.

👨‍⚕️ Use Cases
Predict lung cancer risk for early intervention

Support patient screening in remote clinics

Guide patients with AI-driven advice on health optimization

👨‍💻 Team
Developed by a team of innovators passionate about healthcare + AI:

Sivakumar Balaji

Viswa Bala Bharti

Rubika M

Sabarish sha Kumar
