# import streamlit as st

# import yfinance as yf
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import plotly.express as px
# import requests


# from datetime import date, timedelta
# from prophet import Prophet
# from prophet.plot import plot_plotly
# from plotly import graph_objs as go


# st.title('Stock Dashboard')

# ticker=st.sidebar.text_input('Ticker', 'F')
# today = date.today()
# default_date = today - timedelta(days=111)
# start_date = st.sidebar.date_input("Start Date", default_date)
# end_date = st.sidebar.date_input('End Date')


# data=yf.download(ticker, start=start_date, end=end_date)

# # Function to Fetch Data
# def fetch_data(ticker, start, end):
#     try:
#         data = yf.download(ticker, start=start, end=end)
#         return data
#     except Exception as e:
#         st.write(f"Error fetching data: {e}")
#         return pd.DataFrame()

# # Fetch data for entered ticker
# data = fetch_data(ticker, start_date, end_date)


# def get_ticker (company_name):
#     url = "https://query2.finance.yahoo.com/v1/finance/search"
#     url = url.replace(" ", "%20")
#     user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
#     params = {"q": company_name, "quotes_count": 1, "country": "United States"}

#     res = requests.get(url=url, params=params, headers={'User-Agent': user_agent})
#     data = res.json()

#     company_code = data['quotes'][0]['symbol']
#     return company_code



# st.sidebar.write("To get ticker symbol-")
# company_name = st.sidebar.text_input("Enter the company's name:")
# if company_name:
#     # Fetch and display the company ticker symbol
#     ticker_symbol = get_ticker(company_name)
#     if ticker_symbol:
#         st.sidebar.write(f'The ticker symbol for {company_name} is: {ticker_symbol}')
#     else:
#         st.sidebar.write('No ticker symbol found for the given company name.')


# if data.empty:
#   st.error(f'Ticker "{ticker}" is invalid or data is not available for the given date range.')
# else:
#     fig=px.line(data, x=data.index, y=data['Adj Close'], title=ticker)
#     st.plotly_chart(fig)
#     pricing_data, forecast_data, comparison, news = st.tabs(["**Pricing Data**", "**Forecast Data**", "**Comparison**", "**News**"])
    
#     ##### Pricing data page
#     with pricing_data:
#       st.header(f'Pricing Movements of {ticker}')
    
    
#       def color_df(val):
#         if val>0:
#           color = '#72f292'
#         else:
#           color = '#eb4034'
#         return f'color:{color}'
    
#       data2= data
#       data2['% Change'] = data ['Adj Close']/data ['Adj Close'].shift(1) - 1  
#       st.dataframe(data2.style.applymap(color_df, subset=['% Change']), width=1000, height=400, )
    
    
#       annual_return = data2['% Change'].mean()*252*100
#       st.write('**Annual Return is**',annual_return, '**%**')
#       stdev = np.std(data2['% Change'])*np.sqrt(252)
#       st.write('**Standard Deviation is**',stdev*100, '**%**')
#       st.write('**Risk Adj. Return is**', annual_return/(stdev*100))
    
    
#     ##### Forecast data page
#     with forecast_data:
#       START = "2015-01-01"
#       TODAY = date.today().strftime("%Y-%m-%d")
    
#       n_years = st.slider('**Years of prediction:**', 1, 4)
#       period = n_years * 365
    
    
#       @st.cache_data
#       def load_data(ticker):
#           data = yf.download(ticker, START, TODAY)
#           data.reset_index(inplace=True)
#           return data
    
    
    
#       data_load_state = st.text('Loading data...')
#       data = load_data(ticker)
#       data_load_state.text('Loading data... done!')
    
    
#       st.subheader(f'Raw data of {ticker}')
#       st.dataframe(data.tail(), width=900)
    
    
#       # Plot raw data
#       def plot_raw_data():
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
#         fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
#         fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
#         st.plotly_chart(fig)
     
#       plot_raw_data()
    
    
#       # Predict forecast with Prophet.   
#       df_train = data[['Date','Close']]      #.copy()
#       # df_train['Close'] = df_train['Close'].values.flatten()  # Ensuring it's 1D
#       df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
#       m = Prophet()
#       m.fit(df_train)
#       future = m.make_future_dataframe(periods=period)
#       forecast = m.predict(future)
    
    
#       # Show and plot forecast
#       st.subheader(f'Forecast data of {ticker}')
#       st.write(forecast.tail())
       
#       st.write(f'**Forecast plot for {n_years} year(s)**')
#       fig1 = plot_plotly(m, forecast)
#       st.plotly_chart(fig1)
#       st.write("**In the above plot, ds = datastamp or Date and y = Closing Price**")
    
    
#       st.write("**Forecast components**")
#       fig2 = m.plot_components(forecast)
#       st.write(fig2)
    
#     ##### Comparison
#     with comparison:
#       st.header(f'Stock Market Comparison for {ticker}')
    
#       data=yf.download(ticker, start=start_date, end=end_date)
    
#       st.subheader('Stock Data')
#       st.dataframe(data, width=900)
    
    
#       data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
#       data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
    
#       from sklearn.preprocessing import MinMaxScaler
#       scaler = MinMaxScaler(feature_range=(0,1))
    
    
#       pas_100_days = data_train.tail(100)
#       data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
#       data_test_scale = scaler.fit_transform(data_test)
    
    
#       st.subheader(f'Price of {ticker} vs MA50')
#       ma_50_days = data.Close.rolling(50).mean().values.flatten
#       fig1 = plt.figure(figsize=(8,6))
#       plt.plot(ma_50_days, 'r')
#       plt.plot(data.Close, 'g')
#       plt.show()
#       st.pyplot(fig1)
    
    
#       st.subheader(f'Price of {ticker} vs MA50 vs MA100')
#       ma_100_days = data.Close.rolling(100).mean().values.flatten
#       fig2 = plt.figure(figsize=(8,6))
#       plt.plot(ma_50_days, 'r')
#       plt.plot(ma_100_days, 'b')
#       plt.plot(data.Close, 'g')
#       plt.show()
#       st.pyplot(fig2)
    
    
#       st.subheader(f'Price of {ticker} vs MA100 vs MA200')
#       ma_200_days = data.Close.rolling(200).mean().values.flatten
#       fig3 = plt.figure(figsize=(8,6))
#       plt.plot(ma_100_days, 'r')
#       plt.plot(ma_200_days, 'b')
#       plt.plot(data.Close, 'g')
#       plt.show()
#       st.pyplot(fig3)
    
#     ##### Stock news page
#     import requests
#     import streamlit as st
#     from datetime import datetime
    
#     def get_stock_news(ticker):
#         url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=10"
#         user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
#         response = requests.get(url, headers={'User-Agent': user_agent})
    
#         if response.status_code == 200:
#             data = response.json()
#             news = data.get('news', [])
#             return news
#         else:
#             return []
    
#     def convert_timestamp(unix_timestamp):
#         return datetime.utcfromtimestamp(unix_timestamp).strftime('%d/%m/%Y, %H:%M:%S')
    
#     # Yahoo Finance News Tab
#     with news:
#         st.header(f'Latest News for {ticker}')
        
#         news_articles = get_stock_news(ticker)
    
#         if news_articles:
#             # Loop through and display the news
#             for i, article in enumerate(news_articles[:10]):
#                 st.subheader(f'News {i+1}')
#                 st.write(f"**Title**: {article['title']}")
#                 st.write(f"**Publisher**: {article['publisher']}")
    
#                 # Convert and display the date and time in the format: "dd/mm/yyyy, hour:minute:second"
#                 published_time = convert_timestamp(article['providerPublishTime'])
#                 st.write(f"**Published on**: {published_time}")
                
#                 st.write(f"**Link**: [Read more]({article['link']})")
#                 st.write("---")  # Divider between articles
#         else:
#             st.write(f'No recent news found for {ticker}.')
    

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt

# Streamlit UI
st.title('Stock Dashboard')

# Sidebar inputs
ticker = st.sidebar.text_input('Ticker', 'F')
today = date.today()
default_date = today - timedelta(days=111)
start_date = st.sidebar.date_input("Start Date", default_date)
end_date = st.sidebar.date_input('End Date')

# Function to fetch stock data
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error(f'Ticker "{ticker}" is invalid or data is not available for the given date range.')
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to get ticker symbol for a company
def get_ticker(company_name):
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        params = {"q": company_name, "quotes_count": 1, "country": "United States"}
        res = requests.get(url, params=params, headers={'User-Agent': user_agent})

        # Check if response is successful
        if res.status_code != 200:
            st.error(f"Failed to fetch ticker symbol. Status code: {res.status_code}")
            return None

        data = res.json()
        if 'quotes' in data and data['quotes']:
            company_code = data['quotes'][0]['symbol']
            return company_code
        else:
            st.error(f"No ticker symbol found for the company '{company_name}'.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching ticker symbol: {e}")
        return None
    except ValueError as e:
        st.error(f"JSON decode error: {e}")
        return None

# Sidebar for company name to fetch ticker
company_name = st.sidebar.text_input("Enter the company's name:")
if company_name:
    ticker_symbol = get_ticker(company_name)
    if ticker_symbol:
        st.sidebar.write(f'The ticker symbol for {company_name} is: {ticker_symbol}')
    else:
        st.sidebar.write('No ticker symbol found for the given company name.')

# Fetch data for the entered ticker
data = fetch_data(ticker, start_date, end_date)

# Check if the data is valid before proceeding
if not data.empty:
    # Calculate % Change
    try:
        data['% Change'] = (data['Adj Close'] / data['Adj Close'].shift(1) - 1).values.flatten()
    except Exception as e:
        st.error(f"Error calculating percentage change: {e}")

    # Plotting price movements
    fig = px.line(data, x=data.index, y='Adj Close', title=f'Price Movement of {ticker}')
    st.plotly_chart(fig)

    # Pricing Data Tab
    pricing_data, forecast_data, comparison, news = st.tabs(["**Pricing Data**", "**Forecast Data**", "**Comparison**", "**News**"])
    
    ##### Pricing data page
    with pricing_data:
        st.header(f'Pricing Movements of {ticker}')

        def color_df(val):
            color = '#72f292' if val > 0 else '#eb4034'
            return f'color:{color}'

        st.dataframe(data.style.applymap(color_df, subset=['% Change']), width=1000, height=400)

        annual_return = data['% Change'].mean() * 252 * 100
        st.write('**Annual Return is**', annual_return, '**%**')
        stdev = np.std(data['% Change']) * np.sqrt(252)
        st.write('**Standard Deviation is**', stdev * 100, '**%**')
        st.write('**Risk Adj. Return is**', annual_return / (stdev * 100))

    ##### Forecast data page
    with forecast_data:
        START = "2015-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        n_years = st.slider('**Years of prediction:**', 1, 4)
        period = n_years * 365

        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data

        data_load_state = st.text('Loading data...')
        data = load_data(ticker)
        data_load_state.text('Loading data... done!')

        st.subheader(f'Raw data of {ticker}')
        st.dataframe(data.tail(), width=900)

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        # Predict forecast with Prophet
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader(f'Forecast data of {ticker}')
        st.write(forecast.tail())

        st.write(f'**Forecast plot for {n_years} year(s)**')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)
        st.write("**In the above plot, ds = datestamp or Date and y = Closing Price**")

        st.write("**Forecast components**")
        fig2 = m.plot_components(forecast)
        st.write(fig2)

    ##### Comparison
    with comparison:
        st.header(f'Stock Market Comparison for {ticker}')

        st.subheader('Stock Data')
        st.dataframe(data, width=900)

        data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        pas_100_days = data_train.tail(100)
        data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)

        st.subheader(f'Price of {ticker} vs MA50')
        ma_50_days = data.Close.rolling(50).mean()
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r', label='MA50')
        plt.plot(data.Close, 'g', label='Price')
        plt.legend()
        plt.title(f'Price vs MA50 for {ticker}')
        st.pyplot(fig1)

        st.subheader(f'Price of {ticker} vs MA50 vs MA100')
        ma_100_days = data.Close.rolling(100).mean()
        fig2 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r', label='MA50')
        plt.plot(ma_100_days, 'b', label='MA100')
        plt.plot(data.Close, 'g', label='Price')
        plt.legend()
        plt.title(f'Price vs MA50 vs MA100 for {ticker}')
        st.pyplot(fig2)

        st.subheader(f'Price of {ticker} vs MA100 vs MA200')
        ma_200_days = data.Close.rolling(200).mean()
        fig3 = plt.figure(figsize=(8, 6))
        plt.plot(ma_100_days, 'r', label='MA100')
        plt.plot(ma_200_days, 'b', label='MA200')
        plt.plot(data.Close, 'g', label='Price')
        plt.legend()
        plt.title(f'Price vs MA100 vs MA200 for {ticker}')
        st.pyplot(fig3)

    ##### Stock news page
    with news:
        st.header(f'Latest News for {ticker}')

        def get_stock_news(ticker):
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=10"
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            response = requests.get(url, headers={'User-Agent': user_agent})

            # Check if response is successful
            if response.status_code == 200:
                try:
                    return response.json().get('news', [])
                except ValueError:
                    st.error("Failed to decode JSON response.")
                    return []
            else:
                st.error(f"Error fetching news: {response.status_code}")
                return []

        news_articles = get_stock_news(ticker)

        if news_articles:
            for article in news_articles:
                st.write(f"**Title**: {article['title']}")
                st.write(f"**Published**: {article['providerPublishTime']}")
                st.write(f"**Link**: [Read More]({article['link']})")
        else:
            st.write(f"No news articles found for {ticker}.")

else:
    st.error(f'Ticker "{ticker}" is invalid or data is not available for the given date range.')

