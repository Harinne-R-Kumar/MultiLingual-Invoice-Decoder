from dotenv import load_dotenv
import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
import google.generativeai as genai
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import speech_recognition as sr

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini model and get responses
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, image[0], prompt])
    return response.text

# Function to process uploaded invoice image
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Anomaly detection function
def detect_invoice_anomalies(invoice_data):
    try:
        invoice_data = invoice_data.select_dtypes(include=[np.number])  # Use only numeric columns
        invoice_data.fillna(0, inplace=True)  # Handle missing values
        if invoice_data.shape[1] == 0:  # No numeric columns
            st.sidebar.error("No numeric columns found for anomaly detection.")
            return None

        scaler = StandardScaler()
        invoice_data_scaled = scaler.fit_transform(invoice_data)
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        anomalies = model.fit_predict(invoice_data_scaled)
        invoice_data["Anomaly"] = anomalies

        return invoice_data[invoice_data["Anomaly"] == -1]  # Return detected anomalies
    except Exception as e:
        st.sidebar.error(f"Error in anomaly detection: {str(e)}")
        return None

# Speech recognition function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for speech input...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized speech: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio")
        except sr.RequestError:
            st.error("Could not request results, check your internet connection")
        return ""

# Streamlit UI setup
st.set_page_config(page_title="Multilingual Invoice Decoder", layout="wide")
st.sidebar.header("Invoice Analysis Options")
st.header("Multilingual Invoice Decoder")

input_prompt = """
You are an expert in invoice analysis.
You will receive input images or text data and provide financial insights.
Detect anomalies in invoices and summarize invoice trends.
"""

input_text = st.sidebar.text_input("Input Prompt: ", key="input")
speech_button = st.sidebar.button("Use Speech Input")
if speech_button:
    input_text = recognize_speech()

uploaded_file = st.sidebar.file_uploader("Upload an Invoice (Image or CSV)", type=["jpg", "jpeg", "png", "csv"])
image = ""

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Invoice", use_column_width=True)
    elif uploaded_file.type == "text/csv":
        invoice_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Invoice Data")
        st.dataframe(invoice_df)

        # Perform anomaly detection
        st.sidebar.subheader("Anomaly Detection")
        anomalies = detect_invoice_anomalies(invoice_df)
        if anomalies is not None and not anomalies.empty:
            st.sidebar.write("Detected Anomalies:")
            st.sidebar.dataframe(anomalies)
        else:
            st.sidebar.success("No anomalies detected.")

submit = st.sidebar.button("Analyze Invoice")

if submit:
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_response(input_prompt, image_data, input_text)
        st.subheader("Response from AI")
        st.write(response)
    elif uploaded_file.type == "text/csv":
        st.subheader("Financial Summary & Insights")
        summary = f"Analyzing financial trends from the uploaded invoice dataset: {invoice_df.describe().to_string()}"
        st.write(summary)
