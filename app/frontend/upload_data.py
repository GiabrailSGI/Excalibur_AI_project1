import streamlit as st
import requests
import io
import pandas as pd

def upload_data():
    # Streamlit App
    st.title("Electrical Usage Forecast")

    # Input method selection
    input_type = st.radio("Choose input type:", ("Text Input", "Upload CSV"))

    # Collect user input
    if input_type == "Text Input":
        user_input = st.text_area("Enter your data (CSV format):")
        file = None
    else:
        user_input = None
        file = st.file_uploader("Upload a CSV file", type=["csv"])
         # Check if a file has been uploaded, and trigger a function if so
        if file is not None:
            st.session_state.global_df = pd.read_csv(file)
            try:
                train_model_logic(file)
            except Exception as e:
                st.warning(str(e))

    # Submit button
    if st.button("Forecast"):
        if not user_input and not file:
            st.error("Please provide input data.")
        else:
            # Prepare the request payload
            if file:
                files = {"file": file.getvalue()}
                data = {}
            else:
                files = {}
                data = {"user_input": user_input}

            # Send request to FastAPI backend
            try:
                response = requests.post("http://localhost:8000/forecast/", files=files, data=data)
                response.raise_for_status()
                result = response.json()
                forecast = result.get("forecast", [])
                st.success("Forecast generated successfully!")
                st.write("Forecasted Values:")
                st.write(forecast)
            except requests.exceptions.RequestException as e:
                st.error(f"Error during forecast: {e}")



def train_model_logic(file):
    files = {"file": file.getvalue()}
    response = requests.post("http://localhost:8000/predict/day", files=files)
    response.raise_for_status()
    result = response.json()
    msg = result.get("msg", None)
    st.success("Model trained successfully!")
    st.write(msg)
