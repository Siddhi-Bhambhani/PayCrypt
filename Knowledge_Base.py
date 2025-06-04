import google.generativeai as genai
import streamlit as st
import pandas as pd
import json
import os

# Set your Gemini API Key
GEMINI_API_KEY = "Your API Key"
genai.configure(api_key=GEMINI_API_KEY)

# Knowledge Base File
KB_FILE = "gemini_payment_knowledge_base.json"

# Load knowledge base
def load_knowledge_base():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r") as f:
            return json.load(f)
    return {}

# Save knowledge base
def save_knowledge_base(kb):
    with open(KB_FILE, "w") as f:
        json.dump(kb, f, indent=4)

# Get AI response using Gemini
def get_ai_response(user_query, knowledge_base):
    # Check if query exists in knowledge base
    for question, answer in knowledge_base.items():
        if user_query.lower() in question.lower():
            return f"🔹 Found in Knowledge Base: {answer}"

    # If not found, generate response from Gemini
    prompt = f"""
    You are an AI assistant specialized in payment systems, compliance, and customer support.
    Answer the following user query based on best industry practices and regulatory compliance.

    User Query: {user_query}
    """
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    ai_response = response.text.strip()

    # Save the new question-answer pair
    knowledge_base[user_query] = ai_response
    save_knowledge_base(knowledge_base)

    return f"🤖 AI Response: {ai_response}"

# Streamlit UI
st.title("💳 Payment Knowledge Base Automation")
st.write("Ask any question related to payment systems, compliance, or regulations!")

# Load knowledge base
knowledge_base = load_knowledge_base()

# User input
user_query = st.text_input("🔍 Ask a payment-related question:")

if st.button("Get Answer"):
    if user_query:
        response = get_ai_response(user_query, knowledge_base)
        st.write(response)
    else:
        st.warning("Please enter a question!")

# Display existing knowledge base
if st.checkbox("📖 Show Knowledge Base"):
    st.write(pd.DataFrame(knowledge_base.items(), columns=["Question", "Answer"]))

