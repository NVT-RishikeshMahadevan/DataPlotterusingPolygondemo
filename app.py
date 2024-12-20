import streamlit as st
import pandas as pd
import pandas_market_calendars as mcal
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Polygon API Key
API_KEY = "VwVAwnv9LZ8w1K17a8BLjeqLe1uFPWeH"

def verify_ticker(symbol, data_type):
    """Verify if the ticker exists"""
    if data_type == "Stock":
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    else:
        url = f"https://api.polygon.io/v3/reference/tickers/C:{symbol}"
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(url, headers=headers)
    
    return response.status_code == 200

def get_trading_days(days_lookback):
    nyse = mcal.get_calendar('NYSE')
    end_date = pd.Timestamp.now(tz='US/Eastern')
    start_date = (end_date - pd.Timedelta(days=days_lookback)).tz_convert('US/Eastern')
    trading_days = nyse.valid_days(start_date=start_date.tz_localize(None), end_date=end_date.tz_localize(None))
    return trading_days.tz_convert('US/Eastern')

def fetch_stock_data(ticker, start_date, end_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
    params = {"sort": "asc", "limit": 50000}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json().get('results', [])
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.drop(columns=['t'], inplace=True)
            return df
        else:
            st.warning(f"No data available for Stock Ticker {ticker} in the given date range.")
            return pd.DataFrame()
    else:
        st.error(f"Error fetching stock data for {ticker}: {response.text}")
        return pd.DataFrame()

def fetch_forex_data(pair, start_date, end_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/C:{pair}/range/1/minute/{start_date}/{end_date}"
    params = {"sort": "asc", "limit": 50000}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json().get('results', [])
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.drop(columns=['t'], inplace=True)
            return df
        else:
            st.warning(f"No data available for Forex Pair {pair} in the given date range.")
            return pd.DataFrame()
    else:
        st.error(f"Error fetching forex data for {pair}: {response.text}")
        return pd.DataFrame()

@st.cache_data
def create_candlestick_chart(df, title, show_volume=False):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['o'],
        high=df['h'],
        low=df['l'],
        close=df['c'],
        name='OHLC'
    ))
    
    if show_volume and 'v' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['v'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        fig.update_layout(
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            )
        )
    
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date/Time',
        template='plotly_dark',
        height=600,
        width=1600,
        xaxis_rangeslider_visible=False,
        xaxis={
            'type': 'category',
            'showgrid': True,
            'tickformat': '%Y-%m-%d %H:%M',
            'tickangle': -45,
            'dtick': 60 * 60 * 1000  # Show tick every hour
        },
        yaxis={'showgrid': True}
    )
    return fig
    
def create_candlestick_chart(df, title, show_volume=False):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['o'],
        high=df['h'],
        low=df['l'],
        close=df['c'],
        name='OHLC'
    ))
    
    if show_volume and 'v' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['v'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        fig.update_layout(
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            )
        )
    
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date/Time',
        template='plotly_dark',
        height=600,
        width=3200,
        xaxis_rangeslider_visible=False,
        xaxis={'type': 'category', 'showgrid': True},
        yaxis={'showgrid': True}
    )
    return fig

def fetch_data_in_chunks(data_type, symbol, trading_days):
    df_all = pd.DataFrame()
    total_api_calls = 0
    start_time = time.time()
    progress_bar = st.progress(0)
    
    chunk_size = 52 if data_type == "Stock" else 30
    total_chunks = len(range(0, len(trading_days), chunk_size))
    
    for i, day in enumerate(range(0, len(trading_days), chunk_size)):
        start_date = trading_days[day].date().isoformat()
        end_date = trading_days[min(day + chunk_size, len(trading_days) - 1)].date().isoformat()
        
        if data_type == "Stock":
            df = fetch_stock_data(symbol, start_date, end_date)
        else:
            df = fetch_forex_data(symbol, start_date, end_date)
            
        total_api_calls += 1
        df_all = pd.concat([df_all, df])
        
        elapsed_time = time.time() - start_time
        progress_text = f"Processing {data_type} {symbol} || API Calls: {total_api_calls} || Elapsed Time: {str(timedelta(seconds=int(elapsed_time)))}"
        st.text(progress_text)
        progress_bar.progress((i + 1) / total_chunks)
        
        time.sleep(12)
    
    progress_bar.empty()
    return df_all

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.title = None
    st.session_state.symbol = None
    st.session_state.data_type = None

# Streamlit UI
st.title("Financial Data Explorer")

# Data type selection
data_type = st.radio("Choose Data Type:", ("Forex", "Stock"))

# Manual symbol input with help text
if data_type == "Stock":
    help_text = "Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
else:
    help_text = "Enter forex pair (e.g., EURUSD, GBPUSD, USDJPY)"

symbol = st.text_input("Enter Symbol:", help=help_text).upper()

# Lookback period in days
days_lookback = st.number_input("How many days of historical data?", 
                               min_value=1, max_value=365, value=7)

# Fetch Data Button
if st.button("Fetch Data"):
    if not symbol:
        st.error("Please enter a symbol")
    else:
        # Verify ticker exists
        with st.spinner('Verifying symbol...'):
            if verify_ticker(symbol, data_type):
                trading_days = get_trading_days(days_lookback)
                actual_trading_days = len(trading_days)
                st.info(f"Fetching data for {actual_trading_days} trading days...")
                
                with st.spinner('Fetching data...'):
                    st.session_state.data = fetch_data_in_chunks(data_type, symbol, trading_days)
                    st.session_state.title = f"{symbol} {'Stock Price' if data_type == 'Stock' else 'Exchange Rate'}"
                    st.session_state.symbol = symbol
                    st.session_state.data_type = data_type
                
                if not st.session_state.data.empty:
                    st.success(f"Successfully fetched {len(st.session_state.data)} data points")
            else:
                st.error(f"Symbol not found: {symbol}")

# Display section - only show if we have data
if st.session_state.data is not None and not st.session_state.data.empty:
    with st.expander("View Raw Data"):
        st.dataframe(st.session_state.data)
    
    st.download_button(
        label="Download Data as CSV",
        data=st.session_state.data.to_csv(),
        file_name=f"{st.session_state.symbol}_data.csv",
        mime="text/csv"
    )
    
    dates = pd.Series(st.session_state.data.index.date).unique()
    selected_date = st.selectbox("Select a date to view:", dates)
    
    show_volume = st.checkbox("Show Volume", value=False)
    
    if selected_date:
        daily_data = st.session_state.data[st.session_state.data.index.date == selected_date]
        if not daily_data.empty:
            st.plotly_chart(
                create_candlestick_chart(
                    daily_data, 
                    f"{st.session_state.title} - {selected_date}",
                    show_volume
                ),
                use_container_width=True
            )
    
    st.subheader(f"Full Period Candlestick Chart ({days_lookback} days)")
    st.plotly_chart(
        create_candlestick_chart(
            st.session_state.data, 
            st.session_state.title,
            show_volume
        ),
        use_container_width=True
    )