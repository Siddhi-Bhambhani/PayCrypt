import streamlit as st
import google.generativeai as genai
import speech_recognition as sr

# Configure Google Gemini API
genai.configure(api_key="Your API Key")

def listen():
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎙️ Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        st.write("🔍 Recognizing...")
        query = recognizer.recognize_google(audio)
        st.write(f"🗣️ You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        st.write("❌ Sorry, I couldn't understand.")
        return None
    except sr.RequestError:
        st.write("⚠️ Could not request results, check your internet connection.")
        return None

def get_ai_response(user_query):
    
    try:
        chat_model = genai.GenerativeModel("gemini-1.5-pro")
        response = chat_model.generate_content(user_query)
        return response.text if response else "Sorry, I couldn't process that request."
    except Exception as e:
        return f"Error: {str(e)}"

def voice_assistant():
    
    user_input = listen()
    if user_input:
        ai_response = get_ai_response(user_input)
        st.write(f"🤖 AI Assistant: {ai_response}")

# Streamlit UI
st.title("🧠 AI-Powered Financial Assistant")

st.header("🎙️ AI Voice Assistant")
st.write("Click below to start listening.")

if st.button("▶️ Start Listening", key="start_listening"):
    voice_assistant()
