
import streamlit as st
import requests

from upload_data import upload_data
from stats import stats
from preds import pred_per_month_and_day


if 'global_df' not in st.session_state:
    st.session_state.global_df = None

# Create a navigation sidebar
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ("Upload Data", "Stats", "Prediction Per Day"))

    # Display the page based on user selection
    if page == "Upload Data":
        upload_data()
    elif page == "Stats":
        stats()
    elif page == "Prediction Per Day":
        pred_per_month_and_day()

if __name__ == "__main__":
    main()
