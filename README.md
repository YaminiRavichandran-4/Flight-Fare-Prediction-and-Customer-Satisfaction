# Flight Fare Prediction and Customer Satisfaction

## Project Overview
This project aims to predict flight fare prices and customer satisfaction using machine learning. It involves analyzing historical flight data, including details such as flight routes, dates, and customer reviews, to create predictive models for both fare pricing and customer satisfaction levels. The goal is to develop a comprehensive system that can help airlines optimize pricing strategies and improve customer service.

## Objectives
- Predict flight fares based on various factors such as routes, dates, and class.
- Predict customer satisfaction levels using data such as reviews, service ratings, and flight experiences.
- Build a machine learning model to predict the flight fare price.
- Build a machine learning model to predict customer satisfaction based on flight data.

## Technologies Used
- **Python**: Main programming language for data processing and model development.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning algorithms for regression and classification.
- **Matplotlib / Seaborn**: Data visualization.
- **Streamlit**: For building an interactive web app to predict flight fares and customer satisfaction.
- **Jupyter Notebooks**: For exploratory data analysis and model prototyping.

## Datasets
The project uses publicly available flight data that includes the following features:
- Flight details (airline, source and destination, date of travel, class)
- Historical pricing for flights
- Customer reviews and ratings
- Customer satisfaction scores

## Project Breakdown

### 1. **Flight Fare Prediction Model**
   - **Objective**: Predict the fare of a flight based on multiple features like source, destination, travel date, and class.
   - **Approach**: 
     - Perform data preprocessing and cleaning.
     - Build regression models (such as Random Forest and XGBoost) to predict fare prices.
     - Evaluate model performance using metrics like Mean Squared Error (MSE) and R-squared (R²).
   - **Result**: A trained model capable of predicting flight fares based on given input features.

### 2. **Customer Satisfaction Prediction Model**
   - **Objective**: Predict customer satisfaction (positive/negative) based on flight and service data.
   - **Approach**: 
     - Perform sentiment analysis on customer reviews.
     - Build classification models (e.g., Logistic Regression, Random Forest) to classify customer satisfaction.
     - Evaluate model performance using metrics like accuracy, precision, and recall.
   - **Result**: A trained model capable of predicting whether a customer will be satisfied or dissatisfied based on their review data.

### 3. **Streamlit Web Application**
   - **Objective**: Create an interactive web app for users to input flight details and get predictions for flight fares and customer satisfaction.
   - **Technologies Used**: Streamlit for the front-end web application.
   - **Features**:
     - User inputs flight details (route, date, class).
     - Predict flight fare and customer satisfaction based on input.
   - **Result**: An easy-to-use web application that allows users to interactively get predictions for flight fares and customer satisfaction.

## How to Run the Project Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/YaminiRavichandran-4/Flight-Fare-Prediction-and-Customer-Satisfaction.git
