
# Stock Vision

## Overview

The Stock Dashboard is a web application built using Streamlit that allows users to visualize and analyze stock market data. It provides features such as historical pricing data, forecast predictions, stock comparisons, and the latest news related to the specified stock.

Deployed website: [Stock Vision](https://stock-prediction-5a77cx7hdaxzaywhkqgzbz.streamlit.app/) 

## Features

- **Historical Pricing Data**: Users can view the historical adjusted closing prices for a specified stock ticker.
- **Forecasting**: Using the Prophet library, users can forecast future stock prices based on historical data.
- **Moving Averages Comparison**: Users can compare stock prices against various moving averages (MA50, MA100, MA200).
- **Latest News**: Users can access the latest news articles related to the specified stock ticker.
- **User-Friendly Interface**: The dashboard is designed to be intuitive and easy to navigate.

## Getting Started

### Prerequisites

To run this application, ensure you have the following installed:

- Python 3.7 or higher
- Streamlit
- yfinance
- pandas
- numpy
- matplotlib
- plotly
- Prophet
- requests

### Installation

1. Clone the repository.

2. Install the required packages:

   ```bash
   pip install streamlit yfinance pandas numpy matplotlib plotly prophet requests
   ```

### Running the Application

To start the Stock Dashboard, run the following command in your terminal:

```bash
streamlit run app.py
```

Replace `app.py` with the name of your main application file if it differs.

### Usage

1. Enter a stock ticker symbol in the sidebar (e.g., `AAPL` for Apple Inc.).
2. Specify the date range for historical data.
3. Explore the different tabs: **Pricing Data**, **Forecast Data**, **Comparison**, and **News**.

### Example Screenshots

#### Main Dashboard

![Screenshot (124)](https://github.com/user-attachments/assets/6bf727c2-a063-45d6-b0c6-38a080d75a44)
  <!-- Add a screenshot of the main dashboard here -->

#### Pricing Data Tab

![Screenshot (125)](https://github.com/user-attachments/assets/3ee83d6d-7c03-4b33-89d1-3307956fa3f8)
  <!-- Add a screenshot of the pricing data tab here -->

#### Forecast Data Tab

![Screenshot (127)](https://github.com/user-attachments/assets/bfab9948-3c0b-4e74-96c4-100fa2aa66ff)
  <!-- Add a screenshot of the forecast data tab here -->

#### Comparison Tab

![Screenshot (128)](https://github.com/user-attachments/assets/9985487b-6bf8-4dbb-8762-dccb585ce99f)
  <!-- Add a screenshot of the comparison tab here -->

#### News Tab

![Screenshot (129)](https://github.com/user-attachments/assets/1d735b07-86e2-4270-9968-e2d6e5ef94f3)
  <!-- Add a screenshot of the news tab here -->

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Yahoo Finance API](https://www.yahoofinanceapi.com/)
- [Prophet](https://facebook.github.io/prophet/)
- [Plotly](https://plotly.com/python/)



