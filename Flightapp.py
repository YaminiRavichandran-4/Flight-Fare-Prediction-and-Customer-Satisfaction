import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Preprocessing function for flight fare prediction
def preprocess_input_fare(data):
    # One-hot encode categorical features
    categorical_columns = ['Airline', 'Source', 'Destination', 'Stops']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)
    
    # Ensure alignment with training data columns
    model_columns = [
        'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 
        'Airline_Jet Airways', 'Airline_Jet Airways Business',
        'Source_Delhi', 'Source_Mumbai', 'Source_Bangalore',
        'Source_Kolkata', 'Source_Chennai',
        'Destination_Delhi', 'Destination_Mumbai', 'Destination_Bangalore',
        'Destination_Kolkata', 'Destination_Chennai',
        'Stops_Non-stop', 'Stops_1 Stop', 'Stops_2 Stops', 'Stops_3 Stops',
        'Departure Time', 'Duration'
    ]
    data = data.reindex(columns=model_columns, fill_value=0)
    return data

# Preprocessing function for passenger satisfaction classifier
def preprocess_input_satisfaction(data,training_columns):
    # One-hot encode categorical features
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Ensure alignment with training data columns
    model_columns = [
        'Gender_Male', 'Customer Type_Loyal Customer', 'Type of Travel_Business Travel',
        'Class_Eco', 'Class_Eco Plus', 'Age', 'Flight Distance',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]
    data = data.reindex(columns=model_columns, fill_value=0)
    return data

# Load models
@st.cache_resource
def load_fare_model():
    with open('flight_fare_model.pkl', 'rb') as file:
        model, training_columns = pickle.load(file)
    return model, training_columns

@st.cache_resource
def load_satisfaction_model():
    with open('passenger_satisfaction.pkl', 'rb') as file:
        model, training_columns = pickle.load(file)
    return model,training_columns

# Main App
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a Page", ["Flight Fare Prediction", "Passenger Satisfaction Classifier"])

    if page == "Flight Fare Prediction":
        st.title("Flight Price Prediction App")
        st.write("Provide flight details to get a fare estimate.")

        # User inputs
        col1, col2 = st.columns(2)

        with col1:        
            airline = st.selectbox("Airline", ["Air India", "GoAir", "IndiGo", "Jet Airways", 
                                               "Jet Airways Business", "Multiple carriers", 
                                               "Multiple carriers Premium economy", "SpiceJet", 
                                               "Trujet", "Vistara", "Vistara Premium economy"])
            source = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"])
            departure_date = st.date_input("Departure Date")
            departure_time = st.time_input("Departure Time")

        with col2:
            destination = st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"])
            stops = st.selectbox("Number of Stops", ["Non-stop", "1 Stop", "2 Stops", "3 Stops"])
            duration = st.number_input("Flight Duration (hours)", min_value=0.0, step=0.1)

        # Predict button
        if st.button("Predict Fare"):
            input_data = pd.DataFrame({
                'Airline': [airline],
                'Source': [source],
                'Destination': [destination],
                'Stops': [stops],
                'Departure Date': [departure_date],
                'Departure Time': [departure_time],
                'Duration': [duration]
            })

            try:
                # Load the model and training columns
                model, training_columns = load_fare_model()

                # Preprocess the input and align with training columns
                processed_input = preprocess_input_fare(input_data)
                processed_input = processed_input.reindex(columns=training_columns, fill_value=0)

                # Use only the model for prediction
                prediction = model.predict(processed_input)
                st.success(f"Predicted Flight Fare: ₹{prediction[0]:,.2f}")
            except Exception as e:
                st.error(f"Error in processing input: {e}")

    elif page == "Passenger Satisfaction Classifier":
        st.title("Passenger Satisfaction Prediction")
        st.write("Provide passenger details to predict satisfaction level.")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
            age = st.number_input("Age", min_value=0, max_value=100)
            travel_type = st.selectbox("Type of Travel", ["Personal Travel", "Business Travel"])

        with col2:
            travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
            flight_distance = st.number_input("Flight Distance (km)", min_value=0)
            departure_delay = st.number_input("Departure Delay (minutes)", min_value=0)
            arrival_delay = st.number_input("Arrival Delay (minutes)", min_value=0)

        # Predict button
        if st.button("Predict Satisfaction"):
            input_data = pd.DataFrame({
                'Gender': [gender], 
                'Customer Type': [customer_type],
                'Age': [age],
                'Type of Travel': [travel_type],
                'Class': [travel_class],
                'Flight Distance': [flight_distance],
                'Departure Delay in Minutes': [departure_delay],
                'Arrival Delay in Minutes': [arrival_delay]
            })

            try:
                # Load the satisfaction model
                model,training_columns = load_satisfaction_model()

                # Preprocess the input
                processed_input = preprocess_input_satisfaction(input_data,training_columns)
                processed_input = processed_input.reindex(columns=training_columns, fill_value=0)

                

                # Make predictions
                prediction = model.predict(processed_input)
                satisfaction = "Satisfied" if prediction[0] == 1 else "Neutral or Dissatisfied"
                st.success(f"Predicted Satisfaction: {satisfaction}")
            except Exception as e:
                st.error(f"Error in processing input: {e}")

if __name__ == "__main__":
    main()
