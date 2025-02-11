import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ======================== Load Models ========================
@st.cache_resource
def load_satisfaction_model():
    with open('passenger_satisfaction.pkl', 'rb') as file:
        model, training_columns = pickle.load(file)
    return model, training_columns

@st.cache_resource
def load_fare_prediction_model():
    with open('fare_prediction_model.pkl', "rb") as file:
        model_tuple = pickle.load(file)
    return model_tuple[0], model_tuple[1]  # Model and Feature Names

# ======================== Preprocessing Function for Satisfaction ========================
def preprocess_input_satisfaction(data, training_columns):
    categorical_columns = ['Customer Type', 'Type of Travel', 'Class']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

    # Align with training columns
    data = data.reindex(columns=training_columns, fill_value=0)
    return data

# ======================== Streamlit App ========================
st.title("✈️ Airline Insights: Satisfaction & Fare Prediction")

# Create Tabs
tab1, tab2 = st.tabs(["😊 Passenger Satisfaction", "💰 Flight Fare Prediction"])

# ======================== Passenger Satisfaction Prediction ========================
with tab1:
    st.header("Predict Passenger Satisfaction")

    # User Inputs
    col1, col2 = st.columns(2)

    with col1:
        customer_type = st.selectbox("🛫 Customer Type", ["Loyal Customer", "Disloyal Customer"])
        age = st.number_input("🎂 Age", min_value=0, max_value=100)
        travel_type = st.selectbox("🧳 Type of Travel", ["Personal Travel", "Business Travel"])
        travel_class = st.selectbox("🏷 Class", ["Business", "Eco", "Eco Plus"])
        inflight_wifi = st.number_input("📶 Inflight Wifi Service (0-5)", min_value=0, max_value=5, step=1)
        departure_convenience = st.number_input("🕗 Departure/Arrival Time Convenience (0-5)", min_value=0, max_value=5, step=1)
        ease_booking = st.number_input("💻 Ease of Online Booking (0-5)", min_value=0, max_value=5, step=1)
        food_drink = st.number_input("🍱 Food and Drink (0-5)", min_value=0, max_value=5, step=1)
        online_boarding = st.number_input("🛂 Online Boarding (0-5)", min_value=0, max_value=5, step=1)

    with col2:
        seat_comfort = st.number_input("💺 Seat Comfort (0-5)", min_value=0, max_value=5, step=1)
        inflight_entertainment = st.number_input("🎬 Inflight Entertainment (0-5)", min_value=0, max_value=5, step=1)
        onboard_service = st.number_input("🛎 On-board Service (0-5)", min_value=0, max_value=5, step=1)
        leg_room = st.number_input("🦵 Leg Room Service (0-5)", min_value=0, max_value=5, step=1)
        baggage_handling = st.number_input("🛄 Baggage Handling (0-5)", min_value=0, max_value=5, step=1)
        checkin_service = st.number_input("✅ Check-in Service (0-5)", min_value=0, max_value=5, step=1)
        inflight_service = st.number_input("💼 Inflight Service (0-5)", min_value=0, max_value=5, step=1)
        cleanliness = st.number_input("🧼 Cleanliness (0-5)", min_value=0, max_value=5, step=1)

    # Predict Button
    if st.button("🔍 Predict Satisfaction"):
        input_data = pd.DataFrame({
            'Customer Type': [customer_type],
            'Age': [age],
            'Type of Travel': [travel_type],
            'Class': [travel_class],
            'Inflight wifi service': [inflight_wifi],
            'Departure/Arrival time convenient': [departure_convenience],
            'Ease of Online booking': [ease_booking],
            'Food and drink': [food_drink],
            'Online boarding': [online_boarding],
            'Seat comfort': [seat_comfort],
            'Inflight entertainment': [inflight_entertainment],
            'On-board service': [onboard_service],
            'Leg room service': [leg_room],
            'Baggage handling': [baggage_handling],
            'Checkin service': [checkin_service],
            'Inflight service': [inflight_service],
            'Cleanliness': [cleanliness]
        })

        try:
            model, training_columns = load_satisfaction_model()
            processed_input = preprocess_input_satisfaction(input_data, training_columns)
            prediction = model.predict(processed_input)
            satisfaction = "Satisfied" if prediction[0] == 1 else "Neutral or Dissatisfied"
            st.success(f"😊 Predicted Satisfaction: {satisfaction}")

        except Exception as e:
            st.error(f"⚠ Error: {e}")

# ======================== Flight Fare Prediction ========================
with tab2:
    st.header("Predict Flight Fare")

    # User Inputs
    airline = st.selectbox("✈️ Airline", [
        'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
        'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
        'Trujet', 'Vistara', 'Vistara Premium economy'
    ])

    source = st.selectbox("🌍 Source City", ['Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
    destination = st.selectbox("📍 Destination City", ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata'])
    stops = st.selectbox("🛑 Number of Stops", ['Non-Stop', '1 Stop', '2 Stops', '3 Stops', '4 Stops'])

    journey_date = st.date_input("📆 Journey Date")
    dep_time = st.time_input("⏰ Departure Time")
    arrival_time = st.time_input("⏳ Arrival Time")

    # Process Time Features
    journey_day = journey_date.day
    journey_month = journey_date.month
    dep_hour, dep_min = dep_time.hour, dep_time.minute
    arrival_hour, arrival_min = arrival_time.hour, arrival_time.minute
    duration_hours = arrival_hour - dep_hour
    duration_mins = arrival_min - dep_min

    if duration_mins < 0:
        duration_hours -= 1
        duration_mins += 60

    # Prepare Input
    user_input = {
        "Journey_day": journey_day, "Journey_month": journey_month,
        "dep_hour": dep_hour, "dep_min": dep_min,
        "arrival_hour": arrival_hour, "arrival_min": arrival_min,
        "Duration_hours": duration_hours, "Duration_mins": duration_mins
    }

    # One-Hot Encoding
    model, feature_names = load_fare_prediction_model()
    for col in feature_names:
        user_input[col] = 0  # Default all to 0
    if f"Airline_{airline}" in feature_names:
        user_input[f"Airline_{airline}"] = 1
    if f"Source_{source}" in feature_names:
        user_input[f"Source_{source}"] = 1
    if destination in feature_names:
        user_input[destination] = 1
    if f"Stops_{stops}" in feature_names:
        user_input[f"Stops_{stops}"] = 1

    processed_input = pd.DataFrame([user_input]).reindex(columns=feature_names, fill_value=0)

    if st.button("💰 Predict Fare"):
        predicted_fare = model.predict(processed_input)
        st.success(f"💵 Estimated Flight Fare: ₹{predicted_fare[0]:,.2f}")
