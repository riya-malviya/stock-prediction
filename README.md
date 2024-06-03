# stock-prediction

In this project we build a stock prediction web app in Python using streamlit, yahoo finance, and Facebook Prophet.
This stock prediction project uses streamlit to visualize the prediction of the stock market upto 4 years. It uses libraries like matplotlib, pandas, numpy, streamlit and stocknews to provide latest predictions and news.

**What services provided:**<br/>
-> **Stock Dashboard:** Main dashboard where all the information regarding the ticker, pricing movements and the prediction is shown.<br/><br/>
-> **Pricing Movements:** The pricing movements of ticker including parameters like Close, Open, High, Low, Adj Close, %Change etc. are shown. With annual return and deviations.<br/><br/>
-> **Forecast Data:** This tab shows user the raw data fetched from open source in the form of a time series dataframe which will be used to analyse and predict the future stock prices.<br/><br/>
-> **Forecast Plot:** Now the forecasted data is plotted for the chosen time frame along with the raw data of the particular ticker.<br/><br/>
-> **Forecast Components:** The forecasted data is plotted for daily, weekly and yearly format to give profound understanding to the user about future trends.<br/><br/>
-> Comparison: The third tab is the comparison tab where the price of the selected ticker is compared with MA50(Moving Average), MA100 and MA200.<br/><br/>
-> News: The fourth and last section is the news section where the Top 10 news at that time about the selected ticker are shown with the title and news sentiment.<br/><br/>

**How to run project:**<br/>
-> Run the following commands:<br/>
  ● pip install stocknews<br/>
  ● pip install streamlit<br/>
  ● %%writefile app.py<br/>
  ● ! wget -q -O - ipv4.icanhazip.com<br/>
  ● ! streamlit run app.py & npx localtunnel --port 8501<br/><br/>

-> Now run app.py file<br/><br/>

**Images:**<br/><br/>

<img src="https://github.com/riya-malviya/stock-prediction/assets/171536835/30257bed-036a-4ace-ba13-49333180ee00" width="700" height="400"><br/><br/>
<img src="https://github.com/riya-malviya/stock-prediction/assets/171536835/235d310e-4166-4c10-8ab2-077ec216a05d" width="700" height="400"><br/><br/>
<img src="https://github.com/riya-malviya/stock-prediction/assets/171536835/09d107ea-30bc-47e1-96ae-d745a5602206" width="700" height="400"><br/><br/>
<img src="https://github.com/riya-malviya/stock-prediction/assets/171536835/0696f996-b680-4b44-8b9b-a70a200ec43f" width="700" height="400"><br/><br/>
<img src="https://github.com/riya-malviya/stock-prediction/assets/171536835/29c449bb-0851-4ef2-b9e2-c943c99a2194" width="700" height="400"><br/><br/>
<img src="https://github.com/riya-malviya/stock-prediction/assets/171536835/336e496f-dfac-4f18-be38-7334d5042d3d" width="700" height="400"><br/><br/>
<img src="https://github.com/riya-malviya/stock-prediction/assets/171536835/42a15c7e-46b0-47ed-a044-793598780c6c" width="700" height="400"><br/><br/>
<img src="https://github.com/riya-malviya/stock-prediction/assets/171536835/43130cde-7b0d-44c6-b605-b09f87928892" width="700" height="400"><br/><br/>

