import streamlit as st
import requests

# Function to make an external API call
def call_api(num1, num2):
    """Example external API call that takes two numbers as parameters"""
    # Replace with your actual API URL
    api_url = "http://localhost:8000/predict/day"  # Example URL
    params = {"month": num1, "day": num2}

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()  # Parse the response as JSON
        return data
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def pred_per_month_and_day():
    st.title("Input Month and Day")

    with st.form(key="month_day"):
        num1 = st.number_input("Enter the Month", value=0)
        num2 = st.number_input("Enter the Day", value=0)

        # Submit button
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        # Make the API call when the form is submitted
        st.write(f"You entered: {num1} and {num2}")
        
        result = call_api(num1, num2)

        # Display the result
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.write("API Prediction Response:")
            #st.write(result['data'])
            st.json(result)
