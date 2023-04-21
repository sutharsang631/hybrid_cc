import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Define function to preprocess the data
def preprocess_data(data):
    # Group the data by Driver_ID and sum the relevant columns
    driver_data = data.groupby('Driver_ID')[['Time_Driven', 'Distance', 'Total_Fare', 'rating']].sum()
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    driver_data_scaled = scaler.fit_transform(driver_data)
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(driver_data_scaled)
    # Convert the similarity matrix to a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=driver_data.index, columns=driver_data.index)
    # Sort the drivers by Total_Fare in ascending order
    sorted_drivers = driver_data.sort_values('Total_Fare').index
    return similarity_df, sorted_drivers

# Define the Streamlit app
def app():
    # Create a file uploader
    uploaded_file = st.file_uploader('Upload your dataset', type=['csv'])
    if uploaded_file is not None:
        # Read the dataset
        data = pd.read_csv(uploaded_file)
        # Keep only the relevant columns
        data = data[['Driver_ID', 'Time_Driven', 'Distance', 'Total_Fare', 'rating']]
        # Preprocess the data
        similarity_df, sorted_drivers = preprocess_data(data)
        # Display the recommended drivers with the lowest Total_Fare first
        st.write('Recommended drivers:')
        for driver_id in sorted_drivers:
            st.write(driver_id)
            
if __name__ == '__main__':
    app()