import streamlit as st

import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px


from datetime import date, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


st.title('Stock Dashboard')

ticker=st.sidebar.text_input('Ticker', 'F')
today = date.today()
default_date = today - timedelta(days=111)
start_date = st.sidebar.date_input("Start Date", default_date)
end_date = st.sidebar.date_input('End Date')


data=yf.download(ticker, start=start_date, end=end_date)

import requests

def get_ticker (company_name):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=url, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code

company_name = st.sidebar.text_input("Enter the company's name:")
if company_name:
    # Fetch and display the company ticker symbol
    ticker_symbol = get_ticker(company_name)
    if ticker_symbol:
        st.sidebar.write(f'The ticker symbol for {company_name} is: {ticker_symbol}')
    else:
        st.sidebar.write('No ticker symbol found for the given company name.')

fig=px.line(data, x=data.index, y=data['Adj Close'], title=ticker)
st.plotly_chart(fig)


pricing_data, forecast_data, comparison, news = st.tabs(["**Pricing Data**", "**Forecast Data**", "**Comparison**", "**News**"])

##### Pricing data page
with pricing_data:
  st.header('Pricing Movements')


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


  st.subheader('Raw data')
  st.dataframe(data.tail(), width=900)


  # Plot raw data
  def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 
  plot_raw_data()


  # Predict forecast with Prophet.
  df_train = data[['Date','Close']]
  df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
 

  m = Prophet()
  m.fit(df_train)
  future = m.make_future_dataframe(periods=period)
  forecast = m.predict(future)


  # Show and plot forecast
  st.subheader('Forecast data')
  st.write(forecast.tail())
   
  st.write(f'**Forecast plot for {n_years} year(s)**')
  fig1 = plot_plotly(m, forecast)
  st.plotly_chart(fig1)
  st.write("**In the above plot, ds = datastamp or Date and y = Closing Price**")


  st.write("**Forecast components**")
  fig2 = m.plot_components(forecast)
  st.write(fig2)

##### Comparison
with comparison:
  st.header('Stock Market Comparison')

  data=yf.download(ticker, start=start_date, end=end_date)

  st.subheader('Stock Data')
  st.dataframe(data, width=900)


  data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
  data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler(feature_range=(0,1))


  pas_100_days = data_train.tail(100)
  data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
  data_test_scale = scaler.fit_transform(data_test)


  st.subheader('Price vs MA50')
  ma_50_days = data.Close.rolling(50).mean()
  fig1 = plt.figure(figsize=(8,6))
  plt.plot(ma_50_days, 'r')
  plt.plot(data.Close, 'g')
  plt.show()
  st.pyplot(fig1)


  st.subheader('Price vs MA50 vs MA100')
  ma_100_days = data.Close.rolling(100).mean()
  fig2 = plt.figure(figsize=(8,6))
  plt.plot(ma_50_days, 'r')
  plt.plot(ma_100_days, 'b')
  plt.plot(data.Close, 'g')
  plt.show()
  st.pyplot(fig2)


  st.subheader('Price vs MA100 vs MA200')
  ma_200_days = data.Close.rolling(200).mean()
  fig3 = plt.figure(figsize=(8,6))
  plt.plot(ma_100_days, 'r')
  plt.plot(ma_200_days, 'b')
  plt.plot(data.Close, 'g')
  plt.show()
  st.pyplot(fig3)

##### Stock news page
from stocknews import StockNews
with news:
  st.header(f'News of {ticker}')
  sn = StockNews(ticker, save_news=False)
  df_news = sn.read_rss()
  for i in range(10):
    st.subheader (f'News {i+1}')
    title_sentiment = df_news['sentiment_title'][i]
    news_sentiment = df_news['sentiment_summary'][i]


    s = f"<p style='font-size:20px;'>{df_news['published'][i]}</p>"
    st.markdown(s, unsafe_allow_html=True)
    s2 = f"<p style='font-size:20px;'>{df_news['title'][i]}</p>"
    st.markdown(s2, unsafe_allow_html=True)
    s3 = f"<p style='font-size:20px;'>{df_news['summary'][i]}</p>"
    st.markdown(s3, unsafe_allow_html=True)
    s4 = f"<p style='font-size:20px;'>Title Sentiment {title_sentiment}</p>"
    st.markdown(s4, unsafe_allow_html=True)
    s5 = f"<p style='font-size:20px;'>News Sentiment {news_sentiment}</p>"
    st.markdown(s5, unsafe_allow_html=True)
