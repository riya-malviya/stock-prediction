import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objs as go

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
      st.dataframe(data2.style.applymap(color_df, subset=['% Change']), width=1000, height=400, )
    
    
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
        
        
        # Ticker input
        # ticker = st.text_input('Enter Stock Ticker', 'AAPL')
        
        # Slider for selecting prediction period
        n_years = st.slider('**Years of prediction:**', 1, 4)
        period = n_years * 365
        
        # Load stock data function
        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data
        # st.header(f'Forecast data of {ticker}')
        # n_years = st.slider('**Years of prediction:**', 1, 4)
        # period = n_years * 365
    
        # # Prepare training data with proper date handling
        # def prepare_data_for_prophet(data):
        #     # Ensure 'Date' is properly parsed as datetime
        #     data['Date'] = pd.to_datetime(data.index, errors='coerce')
    
        #     # Drop rows with invalid dates
        #     if data['Date'].isna().sum() > 0:
        #         st.write(f"Found {data['Date'].isna().sum()} invalid dates, removing them.")
        #         data = data.dropna(subset=['Date'])
    
        #     df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
        #     return df_train.dropna()

        # # Load the data
        # def load_data(ticker):
        #     data = yf.download(ticker, start_date, end_date)
        #     return data
    
        data_load_state = st.text('Loading data...')
        data = load_data(ticker)
        data_load_state.text('Loading data... done!')
    
        st.subheader(f'Raw data of {ticker}')
        st.dataframe(data.tail(), width=900)

        # Ensure the 'Date' column is in datetime format
        data['Date'] = pd.to_datetime(data['Date'])


        # Prepare the data for Prophet
        df_train = data[['Date', 'Close']].copy()
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        
        # Create and fit the model
        model = Prophet(daily_seasonality=True)
        model.fit(df_train)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=period)
        
        # Predict future stock prices
        forecast = model.predict(future)
        
        # Show and plot forecast
        st.subheader(f'Forecast data for {ticker}')
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        # Plot forecast
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast)
        
        # Show forecast components (daily, weekly, yearly)
        st.subheader('Forecast Components')
        fig_components = model.plot_components(forecast)
        st.write(fig_components)
    
        # Prepare the data for Prophet
        # df_train = prepare_data_for_prophet(data)
    
        # if df_train.empty:
        #     st.error("Training data is empty after processing.")
        # else:
        #     # Additional check for invalid date values
        #     if df_train['ds'].isna().sum() > 0:
        #         st.error(f"Found invalid date values after processing. Please check your data.")
        #     else:
        #         # Function to fit the Prophet model
        #         def fit_prophet_model(df_train):
        #             m = Prophet()
        #             m.fit(df_train)
        #             return m

        #         # Fit the model and make predictions
        #         model = fit_prophet_model(df_train)
        #         future = model.make_future_dataframe(periods=period)
        #         forecast = model.predict(future)
    
        #         # Show and plot forecast
        #         st.subheader(f'Forecast data of {ticker}')
        #         st.write(forecast.tail())
    
        #         st.write(f'**Forecast plot for {n_years} year(s)**')
        #         fig1 = plot_plotly(model, forecast)
        #         st.plotly_chart(fig1)
    
        #         st.write("**Forecast components**")
        #         fig2 = model.plot_components(forecast)
        #         st.write(fig2)

        
      #   Plot raw data
      #   def plot_raw_data():
      #     fig = go.Figure()
      #     fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
      #     fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
      #     fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
      #     st.plotly_chart(fig)
     
      #   plot_raw_data()
    
    
      # Predict forecast with Prophet.
      #   df_train = data[['Date','Close']]
      #   df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
     
    
      #   m = Prophet()
      #   m.fit(df_train)
      #   future = m.make_future_dataframe(periods=period)
      #   forecast = m.predict(future)
    
    
      # # Show and plot forecast
      #   st.subheader(f'Forecast data of {ticker}')
      #   st.write(forecast.tail())
       
      #   st.write(f'**Forecast plot for {n_years} year(s)**')
      #   fig1 = plot_plotly(m, forecast)
      #   st.plotly_chart(fig1)
      #   st.write("**In the above plot, ds = datastamp or Date and y = Closing Price**")
    
    
      #   st.write("**Forecast components**")
      #   fig2 = m.plot_components(forecast)
      #   st.write(fig2)
    
        ##### Comparison page
    with comparison:
        st.header(f'Stock Market Comparison for {ticker}')
    
        data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])
    
        # Price comparison with moving averages (MA50, MA100, MA200)
        st.subheader(f'Price of {ticker} vs MA50')
        ma_50_days = data.Close.rolling(50).mean()
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r')
        plt.plot(data.Close, 'g')
        plt.show()
        st.pyplot(fig1)
    
        st.subheader(f'Price of {ticker} vs MA50 vs MA100')
        ma_100_days = data.Close.rolling(100).mean()
        fig2 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r')
        plt.plot(ma_100_days, 'b')
        plt.plot(data.Close, 'g')
        plt.show()
        st.pyplot(fig2)
    
        st.subheader(f'Price of {ticker} vs MA100 vs MA200')
        ma_200_days = data.Close.rolling(200).mean()
        fig3 = plt.figure(figsize=(8, 6))
        plt.plot(ma_100_days, 'r')
        plt.plot(ma_200_days, 'b')
        plt.plot(data.Close, 'g')
        plt.show()
        st.pyplot(fig3)
    
        ##### Stock news page
    with news:
        st.header(f'Latest News for {ticker}')
    
        def get_stock_news(ticker):
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=10"
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            response = requests.get(url, headers={'User-Agent': user_agent})
    
            if response.status_code == 200:
                data = response.json()
                return data.get('news', [])
            else:
                return []
    
        def convert_timestamp(unix_timestamp):
            return datetime.utcfromtimestamp(unix_timestamp).strftime('%Y/%m/%d, %H:%M:%S')
    
        news_articles = get_stock_news(ticker)
    
        if news_articles:
            for i, article in enumerate(news_articles[:10]):
                st.subheader(f'News {i + 1}')
                st.write(f"**Title**: {article['title']}")
                st.write(f"**Publisher**: {article['publisher']}")
    
                published_time = convert_timestamp(article['providerPublishTime'])
                st.write(f"**Published on**: {published_time}")
    
                st.write(f"**Link**: [Read more]({article['link']})")
                st.write("---")
        else:
            st.write(f'No recent news found for {ticker}.')
    


