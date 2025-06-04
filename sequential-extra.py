import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import multiprocessing as mp
from plotly import graph_objs as go

def fetch_stock_data(ticker):
    df = yf.download(ticker, start='2015-01-01', end='2025-01-01')
    df.reset_index(inplace=True)
    return df

def parallel_data_fetch(tickers):
    with mp.Pool(mp.cpu_count()) as pool:
        dataframes = pool.map(fetch_stock_data, tickers)
    return dataframes

def main():
    st.title('Stock Forecast Application')

    tickers = st.multiselect('Select Stocks', options=['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA'])

    if st.button('Fetch and Forecast'):
        st.write("Fetching data...")
        stock_data_list = parallel_data_fetch(tickers)

        for ticker, stock_data in zip(tickers, stock_data_list):
            st.write(f'Data for {ticker}')

            stock_data.columns = [col[0] for col in stock_data.columns]
            st.dataframe(stock_data.tail())

            data = stock_data

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)


            # Select and rename the necessary columns
            df_train = data[['Date', 'Close']]  # Adjust 'GOOG' if needed for your specific case
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
            #
            m = Prophet()
            m.fit(df_train)
            #
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)

            st.write(f'Forecast for {ticker}')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.write('Forecast components')
            fig2 = m.plot_components(forecast)
            st.write(fig2)

if __name__ == '__main__':
    main()
