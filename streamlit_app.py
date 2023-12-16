import pickle
import streamlit as st 
import numpy as np
import pandas as pd
from PIL import Image
import json
import requests
from streamlit_lottie import st_lottie
# from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt

from pydantic_settings import BaseSettings# import pandas_profiling
# from pandas_profiling.config import Settings

# Stock
from datetime import date
import yfinance as yf
# from fbprophet import Prophet
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

def home_page():
    st.title("VNDH - Team 27")
    st.markdown("<h3 style='color: #FFA31E;'>Welcome to Team 27 Website!</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#FFA31E;'>Please enter your name and proceed to the next steps.</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #FFA31E;'>You can navigate to your desired page by choosing an option from the drop-down menu on the left.</h3>", unsafe_allow_html=True)
    url = requests.get("https://assets10.lottiefiles.com/packages/lf20_2yyeslc6.json")

    # Creating a blank dictionary to store JSON file,
    # as their structure is similar to Python Dictionary    
    url_json = dict()
    
    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error in the URL")
    
    st_lottie(url_json)

# Template 1: Problem statement & data description
def problem_description_page():
    st.title("Introduction")
    st.markdown("<h1 style='color: #FFA31E;'>Problem Statement</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00EA87;'>Predicting sales in ... and ...</h3>", unsafe_allow_html=True)
    st.write("Our stakeholders, ..., face a business problem related to ..., which result in excessive financial losses. The stakeholders want to ... before ... to allocate costs appropriately.")
    st.write("Our goals are:")
    st.write("1. ")
    st.write("2. ")
    st.write("...")
    st.write("...")
    
    st.markdown("<h3 style='color: #FFA31E;'>Data Description</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Add the text in the first column    
    with col1:
        st.markdown("<span style='color:#00EA87'><b>CustomerID</b></span>: Customer ID", unsafe_allow_html=True)

    # Add the image in the second column    
    with col2:
        image = Image.open("WebApp_Files/Images/Image01.jpg")
        st.image(image, width=700)

# Template 2: Data visualization using Pandas profiling
def data_visualization_page():
    # Read the contents of the CSS file
    with open('style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    st.title("VNDH - Team 27")
    st.markdown("<h3 style='color: #FFA31E;'>Data Visualization using Pandas Profiling</h3>", unsafe_allow_html=True)

    df = pd.read_csv("WebApp_Files/Data/Train_Data.csv")
    profile = ProfileReport(df, title="Pandas Profiling Report")
    st_profile_report(profile)

# Stock
# Template 5: Time series
def time_series():
    with open('style.css') as f:
        css = f.read()
    
    # Add the CSS to the app's page
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    st.subheader(f"Hello! Please select an ...")

    st.write(f"Based on the provided details, we predict that ...")
    # st.subheader(result)
    # st.write("Our predictions have the following metrics:")

    # st.subheader("Accuracy: 94%")
    # st.subheader("Precision: 97%")
    # st.subheader("Recall: 83%")
    # st.subheader("F1 Score: 89%")
    
    @st.cache
    def get_data():
        path = 'WebApp_Files/Data/Train_Data.csv'
        return pd.read_csv(path, low_memory=False)
    
    df = get_data()
    df = df.drop_duplicates(subset="Name", keep="first")
    
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    
    st.title("Sales Prediction")
    st.write("###")
    
    stocks = df['Name']
    # stocks = ("AAPL", "NFLX", "GOOG", "MSFT", "INFY", "RPOWER.NS", "BAJFINANCE.NS", "YESBANK.NS", "RCOM.NS", "EXIDEIND.NS", "TATACHEM.NS", "TATAMOTORS.NS", "RUCHI.NS")
    user_choice = st.selectbox("Select dataset and years for prediction", stocks)
    
    index = df[df["Name"]==user_choice].index.values[0]
    symbol = df["Symbol"][index]
    
    n_years = st.slider("", 1, 5)
    period = n_years * 365
    
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data
    
    data_load_state = st.text("Load data ...")
    data = load_data(symbol)
    data_load_state.text("Loading data ... Done!")
    
    st.write("###")
    
    st.subheader("Raw data")
    st.write(data.tail())
    
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
        st.plotly_chart(fig)
    
    plot_raw_data()
    
    # Forecasting
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
    m = Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.write("***")
    st.write("###")
    
    st.subheader("Forecast data")
    st.write(forecast.tail())
    
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    st.subheader("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)



def main():
    """
    This function defines the main function to run the web page.
    """    
    st.set_page_config(page_title="VNDH - Team 27", page_icon=":guardsman:",
                       layout="wide", initial_sidebar_state="expanded")

    with open('style.css') as f:
        css = f.read()

    # Add the CSS to the app's page
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    # Define the sidebar menu
    menu = ["Home", "Introduction", "Data Overview", "Sales Prediction"]

    # Add input fields for name and details in the sidebar
    custom_css = """
        <style>
            .dropdown-menu { 
                background-color: white !important;
            }
        </style>
    """
    
    with st.sidebar:
        st.subheader("User Information")
        name = st.text_input("User Name")

    # Show the appropriate page based on the user's choice
        st.sidebar.subheader("Select Your Choice")
        
        choice = st.sidebar.selectbox("Select a page", menu)

        url = requests.get("https://assets8.lottiefiles.com/packages/lf20_goa8injd.json")
        # Creating a blank dictionary to store JSON file,
        # as their structure is similar to Python Dictionary        
        url_json = dict()

        if url.status_code == 200:
            url_json = url.json()
        else:
            print("Error in the URL")
    
        st_lottie(url_json)

    if choice == "Home":
        home_page()
    elif choice == "Introduction":
        problem_description_page()
    elif choice == "Data Overview":
        data_visualization_page()
    elif choice == "Sales Prediction":
        time_series()



if __name__ == "__main__":
    main()
