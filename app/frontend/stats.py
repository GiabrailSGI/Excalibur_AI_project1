

import streamlit as st 
import matplotlib.pyplot as plt
import pandas as pd

def stats():
    dataframe = st.session_state.global_df
    if dataframe is None:
        st.write("No dataframe provided")
    else:
        data = dataframe
        print(data.columns)
        print(data.head())

        # Combine the Date and Time columns into a single DateTime column
        data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

        # Drop rows with missing DateTime values
        data.dropna(subset=['DateTime'], inplace=True)

        # Convert relevant columns to numeric
        numeric_columns = [
            'Global_active_power', 'Global_reactive_power', 'Voltage',
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
        ]
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Drop rows with missing values in the numeric columns
        data.dropna(subset=numeric_columns, inplace=True)

        # Streamlit app
        st.title("Household Power Consumption Analysis")

        # Display dataset
        if st.checkbox("Show raw data"):
            st.write(data.head())

        # Statistic 1: Summary statistics
        st.header("Summary Statistics")
        st.write(data[numeric_columns].describe())

        # Statistic 2: Global Active Power distribution
        st.header("Global Active Power Distribution")
        fig, ax = plt.subplots()
        data['Global_active_power'].plot(kind='hist', bins=50, ax=ax, color='blue', alpha=0.7)
        ax.set_title("Distribution of Global Active Power")
        ax.set_xlabel("Global Active Power (kW)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Statistic 3: Voltage over time
        st.header("Voltage Over Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        data.set_index('DateTime')['Voltage'].plot(ax=ax, color='green')
        ax.set_title("Voltage Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Voltage (V)")
        st.pyplot(fig)

        # Statistic 4: Correlation heatmap
        st.header("Correlation Heatmap")
        import seaborn as sns
        fig, ax = plt.subplots()
        correlation_matrix = data[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        # Statistic 5: Average Global Intensity by Hour
        st.header("Average Global Intensity by Hour")
        data['Hour'] = data['DateTime'].dt.hour
        avg_intensity = data.groupby('Hour')['Global_intensity'].mean()
        fig, ax = plt.subplots()
        avg_intensity.plot(kind='bar', ax=ax, color='purple', alpha=0.7)
        ax.set_title("Average Global Intensity by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Global Intensity (A)")
        st.pyplot(fig)

        # Global Active Power vs Voltage scatter plot
        st.header("Global Active Power vs Voltage")
        fig, ax = plt.subplots()
        ax.scatter(data['Voltage'], data['Global_active_power'], alpha=0.5, color='orange')
        ax.set_title("Global Active Power vs Voltage")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Global Active Power (kW)")
        st.pyplot(fig)