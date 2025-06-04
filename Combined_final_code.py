import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import pandas as pd
import json
import os
import numpy as np
import random
import time
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from langchain_google_genai import ChatGoogleGenerativeAI

import warnings


warnings.filterwarnings("ignore")

GEMINI_API_KEY = ("Your API Key")
genai.configure(api_key=GEMINI_API_KEY)


gateways = ["Stripe", "PayPal", "Razorpay"]
languages = ["Python", "Node.js", "Java"]


KB_FILE = "gemini_payment_knowledge_base.json"
def load_knowledge_base():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_knowledge_base(kb):
    with open(KB_FILE, "w") as f:
        json.dump(kb, f, indent=4)
st.markdown(
    """
    <style>
        /* Full background gradient */
        .stApp {
            background: linear-gradient(to right, #00FFFF, #008B8B);
            color: white;
        }

        /* Tabs container */
        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(0, 102, 102, 0.9) !important;
            border-radius: 15px;
            padding: 8px;
        }

        /* Non-active tab styling */
        .stTabs [data-baseweb="tab"] {
            color: white !important;
            font-size: 18px !important;
            font-weight: bold;
            padding: 10px 20px !important;
            border-radius: 10px !important;
            background-color: rgba(0, 100, 100, 0.5) !important;
        }

        /* Active tab styling */
        .stTabs [aria-selected="true"] {
            background-color: #00CED1 !important;
            color: black !important;
            font-weight: bold;
            padding: 12px 24px !important;
            border-radius: 10px;
        }

        /* Styling input fields */
        input, textarea {
            background-color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }

        /* Improve button visibility */
        button {
            background-color: #009999 !important;
            border-radius: 10px !important;
            color: white !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }

        /* Header styling */
        h1, h2, h3 {
            font-weight: bold !important;
            color: #033B4A !important;
        }
        
        
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def generate_synthetic_data(n=10000):
    data = []
    for _ in range(n):
        amount = round(random.uniform(10, 10000), 2)
        frequency = random.randint(1, 20)
        device_type = random.choice(["mobile", "desktop", "tablet"])
        country = random.choice(["US", "IN", "UK", "CA", "DE", "CN"])
        time_of_day = random.randint(0, 23)
        fraud = random.choices([0, 1], weights=[0.98, 0.02])[0]
        data.append([amount, frequency, device_type, country, time_of_day, fraud])
    return pd.DataFrame(data, columns=["amount", "frequency", "device_type", "country", "time_of_day", "fraud"])

df = generate_synthetic_data()
df["device_type"] = df["device_type"].astype("category").cat.codes
df["country"] = df["country"].astype("category").cat.codes
features = ["amount", "frequency", "device_type", "country", "time_of_day"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(df[features])

def get_risk_score(transaction):
    transaction_scaled = scaler.transform([transaction])
    anomaly_score = model.decision_function(transaction_scaled)[0]
    return round(max(0, min(100, (1 - anomaly_score) * 70 + random.uniform(0, 30))), 2)

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎙️ Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        return query.lower()
    except:
        return None

def get_ai_response(user_query, knowledge_base):
    for question, answer in knowledge_base.items():
        if user_query.lower() in question.lower():
            return f"🔹 Found in Knowledge Base: {answer}"
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(user_query)
    ai_response = response.text.strip()
    knowledge_base[user_query] = ai_response
    save_knowledge_base(knowledge_base)
    return f"🤖 AI Response: {ai_response}"

st.title("💳 AI-Powered Financial Tools")
tabs = st.tabs(["Code Generator", "Voice Assistant", "Fraud Detection", "Knowledge Base"])

with tabs[0]:
    st.header("🔧 AI-Powered Payment Integration Code Generator")
    gateway = st.selectbox("Select Payment Gateway:", gateways)
    language = st.selectbox("Select Programming Language:", languages)
    custom_request = st.text_area("Custom Instructions (Optional)", "")
    if st.button("Generate Code"):
        model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
        prompt = f"Generate a {language} code snippet for integrating {gateway} payment gateway. {custom_request}"
        response = model.invoke(prompt)
        st.write(response.content)

with tabs[1]:
    st.header("🧠 AI Voice Assistant")
    if st.button("▶️ Start Listening"):
        user_input = listen()
        if user_input:
            st.write(f"🗣️ You said: {user_input}")
            ai_response = get_ai_response(user_input, load_knowledge_base())
            st.write(f"🤖 AI Assistant: {ai_response}")

with tabs[2]:
    st.header("🔍 Real-Time Fraud Detection System")
    # Initialize session state variables
    if "running" not in st.session_state:
        st.session_state.running = False
    if "transactions" not in st.session_state:
        st.session_state.transactions = pd.DataFrame(columns=["Amount", "Frequency", "Device", "Country", "Time", "Risk Score", "Alert"])
    if "fraud_count" not in st.session_state:
        st.session_state.fraud_count = 0

    # Buttons to control monitoring
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Monitoring"):
            st.session_state.running = True
            st.session_state.transactions = pd.DataFrame(columns=["Amount", "Frequency", "Device", "Country", "Time", "Risk Score", "Alert"])
            st.session_state.fraud_count = 0

    with col2:
        if st.button("🛑 Stop Monitoring"):
            st.session_state.running = False

    # Display transactions in a table
    transaction_table = st.empty()

    # Monitor Transactions
    while st.session_state.running:
        new_transaction = [
            round(random.uniform(10, 10000), 2),  # amount
            random.randint(1, 20),  # frequency
            random.choice(["Mobile", "Desktop", "Tablet"]),  # device_type
            random.choice(["US", "IN", "UK", "CA", "DE", "CN"]),  # country
            random.randint(0, 23)  # time_of_day
        ]
        
        risk_score = get_risk_score([new_transaction[0], new_transaction[1], random.randint(0, 2), random.randint(0, 5), new_transaction[4]])
        alert = "🚨 FRAUD ALERT" if risk_score > 90 else "✅ Safe"

        if alert == "🚨 FRAUD ALERT":
            st.session_state.fraud_count += 1
        


        # Add transaction to dataframe
        new_entry = pd.DataFrame([new_transaction + [risk_score, alert]], columns=["Amount", "Frequency", "Device", "Country", "Time", "Risk Score", "Alert"])
        st.session_state.transactions = pd.concat([new_entry, st.session_state.transactions], ignore_index=True).head(50)  # Keep last 50 transactions

        transaction_table.dataframe(st.session_state.transactions, height=500, use_container_width=True)

        time.sleep(1)  # Simulate real-time delay

    # Show summary when monitoring stops
    if not st.session_state.running and not st.session_state.transactions.empty:
        st.write(f"📊 **Total Fraud Alerts:** {st.session_state.fraud_count}")

with tabs[3]:
    st.header("📖 Payment Knowledge Base Automation")
    knowledge_base = load_knowledge_base()
    user_query = st.text_input("🔍 Ask a payment-related question:")
    if st.button("Get Answer"):
        if user_query:
            response = get_ai_response(user_query, knowledge_base)
            st.write(response)
        else:
            st.warning("Please enter a question!")
    if st.checkbox("📖 Show Knowledge Base"):
        st.write(pd.DataFrame(knowledge_base.items(), columns=["Question", "Answer"]))
