import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Load the dataset
def load_data():
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "txt"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df

# Select the column to base the recommendation on
def select_column(df):
    columns = ["trip id", "Driver_ID", "Time_Driven", "Distance", "Total_Fare", "rating", "experience"]
    selected_column = st.selectbox("Select the column to base the recommendation on", columns)
    return selected_column

# Use a hybrid recommendation algorithm to recommend drivers
def recommend_drivers(df, column):
    # Convert the dataframe to the Surprise dataset format
    reader = Reader(rating_scale=(df[column].min(), df[column].max()))
    data = Dataset.load_from_df(df[['Driver_ID', 'rating', column]], reader)
    
    # Use SVD for collaborative filtering
    algo = SVD()
    
    # Use cross-validation to evaluate the algorithm
    cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)
    
    # Fit the algorithm on the entire dataset
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    
    # Get the predictions for all drivers
    drivers = df['Driver_ID'].unique()
    predictions = []
    for driver in drivers:
        rating = df[df['Driver_ID'] == driver]['rating'].values[0]
        prediction = algo.predict(driver, rating)
        predictions.append((driver, prediction.est))
    
    # Sort the drivers based on their predicted scores
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return the recommended drivers
    return [x[0] for x in predictions]

# Main function
def main():
    st.title("Driver Recommendation App")
    st.subheader("Select the dataset and the column to base the recommendation on")
    try:
        # Load the dataset
        df = load_data()
        
        # Select the column to base the recommendation on
        column = select_column(df)
        
        # Recommend the best drivers
        drivers = recommend_drivers(df, column)
        # Display the recommended drivers to the user
        st.subheader("Recommended drivers ID")
        for row in drivers:
        
            st.write( "Driver_ID:  ",row)
    except ValueError:
        st.write('string type column are not allowed')
    except :
        st.write('upload the document')

if __name__ == "__main__":
    main()
