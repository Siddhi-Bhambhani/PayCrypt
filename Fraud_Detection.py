import streamlit as st
import numpy as np
import pandas as pd
import random
import time
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore")


@st.cache_data
def generate_synthetic_data(n=10000):
    data = []
    for _ in range(n):
        amount = round(random.uniform(10, 10000), 2)  # Transaction amount
        frequency = random.randint(1, 20)  # Transactions per week
        device_type = random.choice(["mobile", "desktop", "tablet"])
        country = random.choice(["US", "IN", "UK", "CA", "DE", "CN"])
        time_of_day = random.randint(0, 23)  # Hour of transaction
        fraud = random.choices([0, 1], weights=[0.98, 0.02])[0]  # 2% fraud rate
        
        data.append([amount, frequency, device_type, country, time_of_day, fraud])
    
    df = pd.DataFrame(data, columns=["amount", "frequency", "device_type", "country", "time_of_day", "fraud"])
    return df

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

    risk_score = max(0, min(100, (1 - anomaly_score) * 70 + random.uniform(0, 30)))  # Adjust variation
    return round(risk_score, 2)


st.title("🔍 Real-Time Fraud Detection System")

st.write("Click **Start Monitoring** to begin tracking transactions in real-time.")


if "running" not in st.session_state:
    st.session_state.running = False
if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=["Amount", "Frequency", "Device", "Country", "Time", "Risk Score", "Alert"])
if "fraud_count" not in st.session_state:
    st.session_state.fraud_count = 0


col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Start Monitoring"):
        st.session_state.running = True
        st.session_state.transactions = pd.DataFrame(columns=["Amount", "Frequency", "Device", "Country", "Time", "Risk Score", "Alert"])
        st.session_state.fraud_count = 0

with col2:
    if st.button("🛑 Stop Monitoring"):
        st.session_state.running = False


transaction_table = st.empty()

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

    
    new_entry = pd.DataFrame([new_transaction + [risk_score, alert]], columns=["Amount", "Frequency", "Device", "Country", "Time", "Risk Score", "Alert"])
    st.session_state.transactions = pd.concat([new_entry, st.session_state.transactions], ignore_index=True).head(50)  # Keep last 50 transactions

    transaction_table.dataframe(st.session_state.transactions, height=500, use_container_width=True)

    time.sleep(1)  


if not st.session_state.running and not st.session_state.transactions.empty:
    st.write(f"📊 **Total Fraud Alerts:** {st.session_state.fraud_count}")
