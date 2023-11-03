# import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Define poly globally
poly = PolynomialFeatures(degree=4)

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to create and train the model
def create_model(data):
    df = data.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].map(datetime.toordinal)

    X = df[['Date']]
    y = df['Close']

    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

# Function to make predictions
def predict_price(model, date):
    date_ordinal = datetime.strptime(date, '%Y-%m-%d').toordinal()
    date_ordinal_poly = poly.transform([[date_ordinal]])

    return model.predict(date_ordinal_poly)

# Streamlit web app
def main():
    st.title('Stock Price Prediction App')
    start_date = st.date_input('Start date', datetime.now() - timedelta(365))
    end_date = st.date_input('End date', datetime.now())

    ticker = st.text_input('Enter Stock Ticker', 'AAPL')
    data = get_stock_data(ticker, start_date, end_date)

    if data.empty:
        st.warning('No data found for the entered stock symbol. Please try another symbol.')
    else:
        st.write(data)

        model = create_model(data)

        selected_date = st.date_input('Select a date to predict the stock price', datetime.now())
        if st.button('Predict'):
            predicted_price = predict_price(model, selected_date.strftime('%Y-%m-%d'))
            st.write('Predicted Stock Price: $', round(predicted_price[0], 2))

if __name__ == '__main__':
    main()
