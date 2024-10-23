import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import requests
import feedparser
from datetime import date, timedelta, datetime
from prophet import Prophet
from prophet.plot import plot_plotly
from bs4 import BeautifulSoup



st.title('Stock Dashboard')

# Sidebar input for ticker and date range
ticker = st.sidebar.text_input('Ticker', 'AAPL')
today = date.today()
default_date = today - timedelta(days=111)
default_end_date = today - timedelta(days=1)
start_date = st.sidebar.date_input("Start Date", default_date)
end_date = st.sidebar.date_input("End Date", default_end_date)

# Fetch stock data from Yahoo Finance
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.write(f"Error fetching data: {e}")
        return pd.DataFrame()

data = fetch_data(ticker, start_date, end_date)



# Check if data is empty
if data.empty:
    st.error(f'Ticker "{ticker}" is invalid or data is not available for the given date range.')
else:
    fig = px.line(data, x=data.index, y=data['Adj Close'].values.flatten(), title=ticker)
    st.plotly_chart(fig)
    pricing_data, forecast_data, comparison, news = st.tabs(["**Pricing Data**", "**Forecast Data**", "**Comparison**", "**News**"])

        ##### Pricing data page
    with pricing_data:
      st.header(f'Pricing Movements of {ticker}')
    
    
      def color_df(val):
        if val>0:
          color = '#72f292'
        else:
          color = '#eb4034'
        return f'color:{color}'
    
    
      data2= data
      data2['% Change'] = data ['Adj Close']/data ['Adj Close'].shift(1) - 1
      # st.dataframe(data2.style.applymap(color_df, subset=['% Change']), width=1000, height=400)
      st.dataframe(data2.style.map(color_df, subset=['% Change']), width=1000, height=400)
    
    
      annual_return = data2['% Change'].mean()*252*100
      st.write('**Annual Return is**',annual_return, '**%**')
      stdev = np.std(data2['% Change'])*np.sqrt(252)
      st.write('**Standard Deviation is**',stdev*100, '**%**')
      st.write('**Risk Adj. Return is**', annual_return/(stdev*100))
    
    ##### Forecast data page
    with forecast_data:
        # Set date range
      START = "2015-01-01"
      TODAY = date.today().strftime("%Y-%m-%d")
        
        
        # Load stock data function
      @st.cache_data
      def load_data(ticker):
          data = yf.download(ticker, START, TODAY)
          data.reset_index(inplace=True)
          return data

                # Slider for selecting prediction period
      st.header(f'Forecast data of {ticker}')
      n_years = st.slider('**Years of prediction:**', 1, 4)
      period = n_years * 365
    
      data_load_state = st.text('Loading data...')
      data = load_data(ticker)
      data_load_state.text('Loading data... done!')
    
      st.subheader(f'Raw data of {ticker}')
      st.dataframe(data.tail(), width=900)

        # Flatten the multi-index columns
      data.columns = ['_'.join(col).strip() for col in data.columns.values]
    
    # Check the new column names (you can print this if needed)
      print(data.columns)
    
    # Rename columns (adjust based on the actual column names after flattening)
      data_train = data.rename(columns={"Date_": "ds", f"Close_{ticker}": "y"})
    
    # Check for missing values or invalid data in 'y' column
      data_train = data_train.dropna(subset=['y'])  # Remove rows with missing 'y' values
      data_train['y'] = pd.to_numeric(data_train['y'], errors='coerce')  # Ensure 'y' is numeric
      data_train = data_train.dropna(subset=['y'])  # Remove rows with invalid 'y' values
      data_train['ds'] = pd.to_datetime(data_train['ds']).dt.tz_localize(None)
    
    # Create and fit the Prophet model
      model = Prophet()
      model.fit(data_train)

      future = model.make_future_dataframe(periods=period)
        
        # Predict future stock prices
      forecast = model.predict(future)
        
        # Show and plot forecast
      st.subheader(f'Forecast data for {ticker}')
      st.write(forecast.tail())
        
        # Plot forecast
      fig_forecast = plot_plotly(model, forecast)
      st.plotly_chart(fig_forecast)
        
        # Show forecast components (daily, weekly, yearly)
      st.subheader('Forecast Components')
      fig_components = model.plot_components(forecast)
      st.write(fig_components)
  
    
        ##### Comparison page
    with comparison:
        st.header(f'Stock Market Comparison for {ticker}')
    
        data_train = pd.DataFrame(data[f"Close_{ticker}"][0:int(len(data) * 0.80)])
        data_test = pd.DataFrame(data[f"Close_{ticker}"][int(len(data) * 0.80):])
    
        # Price comparison with moving averages (MA50, MA100, MA200)
        st.subheader(f'Price of {ticker} vs MA50')
        ma_50_days = data[f"Close_{ticker}"].rolling(50).mean()
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r')
        plt.plot(data[f"Close_{ticker}"], 'g')
        plt.show()
        st.pyplot(fig1)
    
        st.subheader(f'Price of {ticker} vs MA50 vs MA100')
        ma_100_days = data[f"Close_{ticker}"].rolling(100).mean()
        fig2 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r')
        plt.plot(ma_100_days, 'b')
        plt.plot(data[f"Close_{ticker}"], 'g')
        plt.show()
        st.pyplot(fig2)
    
        st.subheader(f'Price of {ticker} vs MA100 vs MA200')
        ma_200_days = data[f"Close_{ticker}"].rolling(200).mean()
        fig3 = plt.figure(figsize=(8, 6))
        plt.plot(ma_100_days, 'r')
        plt.plot(ma_200_days, 'b')
        plt.plot(data[f"Close_{ticker}"], 'g')
        plt.show()
        st.pyplot(fig3)
    
        #### Stock news page
        def get_stock_news(ticker):
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=10"
            
            
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            response = requests.get(url, headers={'User-Agent': user_agent})
        
            if response.status_code == 200:
                data = response.json()
                news = data.get('news', [])
                return news
            else:
                return []
    
        def convert_timestamp(unix_timestamp):
            return datetime.utcfromtimestamp(unix_timestamp).strftime('%d/%m/%Y, %H:%M:%S')
        
        # Yahoo Finance News Tab
        with news:
            st.header(f'Latest News for {ticker}')
            
            news_articles = get_stock_news(ticker)
        
            if news_articles:
                # Loop through and display the news
                for i, article in enumerate(news_articles[:10]):
                    st.subheader(f'News {i+1}')
                    st.write(f"**Title**: {article['title']}")
                    st.write(f"**Publisher**: {article['publisher']}")
        
                    # Convert and display the date and time in the format: "dd/mm/yyyy, hour:minute:second"
                    published_time = convert_timestamp(article['providerPublishTime'])
                    st.write(f"**Published on**: {published_time}")
                    
                    st.write(f"**Link**: [Read more]({article['link']})")
                    st.write("---")  # Divider between articles
            else:
                st.write(f'No recent news found for {ticker}.')
    
            
        # import requests
        # from datetime import datetime
        # import streamlit as st
        
        # # Function to get stock news
        # def get_stock_news(ticker):
        #     url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=10"
        #     user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            
        #     response = requests.get(url, headers={'User-Agent': user_agent})
            
        #     if response.status_code == 200:
        #         data = response.json()
        #         news = data.get('news', [])
        #         return news
        #     else:
        #         return []
        
        # # Function to convert UNIX timestamp to human-readable format
        # def convert_timestamp(unix_timestamp):
        #     return datetime.utcfromtimestamp(unix_timestamp).strftime('%d/%m/%Y, %H:%M:%S')
        
        # # Yahoo Finance News Tab in Streamlit
        # with news:
        
        #     if ticker:
        #         st.header(f'Latest News for {ticker}')
                
        #         # Get news articles
        #         news_articles = get_stock_news(ticker)
                
        #         if news_articles:
        #             # Loop through and display the news
        #             for i, article in enumerate(news_articles):
        #                 title = article.get('title', 'No Title')
        #                 publisher = article.get('publisher', 'Unknown Publisher')
        #                 published_time = article.get('providerPublishTime', 0)
        #                 link = article.get('link', '#')
                        
        #                 # Convert the timestamp to human-readable format
        #                 if published_time:
        #                     published_time = convert_timestamp(published_time)
        #                 else:
        #                     published_time = "Unknown Time"
                        
        #                 st.subheader(f'News {i+1}')
        #                 st.write(f"**Title**: {title}")
        #                 st.write(f"**Publisher**: {publisher}")
        #                 st.write(f"**Published on**: {published_time}")
        #                 st.write(f"**Link**: [Read more]({link})")
        #                 st.write("---")  # Divider between articles
        #         else:
        #             st.write(f'No recent news found for {ticker}.')

        



       
