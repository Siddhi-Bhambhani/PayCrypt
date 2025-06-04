import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
# Define payment gateways & languages
gateways = ["Stripe", "PayPal", "Razorpay"]
languages = ["Python", "Node.js", "Java"]

# Streamlit UI
st.title("🔧 AI-Powered Payment Integration Code Generator")
st.write("Select a payment gateway and programming language to generate integration code.")

# User Inputs
gateway = st.selectbox("Select Payment Gateway:", gateways)
language = st.selectbox("Select Programming Language:", languages)
custom_request = st.text_area("Custom Instructions (Optional)", "")

if st.button("Generate Code"):
    # Prompt for Gemini API
    prompt = f"Generate a {language} code snippet for integrating {gateway} payment gateway. Include API authentication, payment processing, and success handling. {custom_request}"
    
    try:
        # Call Gemini via LangChain
        response =model.invoke(prompt)
        
        # Display generated code
        st.write(response.content)
    except Exception as e:
        st.error(f"Error: {e}")

st.write("👨‍💻 Supports Stripe, PayPal, and Razorpay. More integrations coming soon!")
