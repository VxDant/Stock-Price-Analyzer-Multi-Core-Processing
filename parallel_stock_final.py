import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from plotly import graph_objs as go



# Define a function to fetch stock data for a given ticker
def fetch_stock_data(ticker):
    try:
        df = yf.download(ticker, start='2015-01-01', end='2025-01-01')
        df.reset_index(inplace=True)
        return ticker, df
    except Exception as e:
        return ticker, pd.DataFrame()

# Parallel data fetching with ProcessPoolExecutor for better scalability
def parallel_data_fetch(tickers):
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = executor.map(fetch_stock_data, tickers)
    return list(results)

# Parallelized data cleaning step for stock data
def clean_data(df):
    df.columns = [col[0] for col in df.columns]
    if df.empty:
        return df
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def main():
    st.title('Stock Forecast Application')

    tickers = st.multiselect('Select Stocks', options=['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA'])

    if st.button('Fetch and Forecast'):
        st.write("Fetching data...")
        stock_data_list = parallel_data_fetch(tickers)

        for ticker, stock_data in stock_data_list:
            if stock_data.empty:
                st.write(f"No data fetched for {ticker}")
                continue

            stock_data = clean_data(stock_data)
            st.write(f'Data for {ticker}')
            st.dataframe(stock_data.tail())

            # Time series visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

            # Prepare data for forecasting
            df_train = stock_data[['Date', 'Close']]
            df_train.columns = ['ds', 'y']


            # Forecasting using Prophet in parallel if multiple tickers
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                future = executor.submit(run_forecast, df_train)
                forecast, model = future.result()

            st.write(f'Forecast for {ticker}')
            st.dataframe(forecast.tail())
            st.plotly_chart(plot_plotly(model, forecast))

# Separate forecasting function to allow parallel execution
def run_forecast(df_train):
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast, model

if __name__ == '__main__':
    main()
