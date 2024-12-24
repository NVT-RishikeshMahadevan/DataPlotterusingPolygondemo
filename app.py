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
    elif data_type == "Crypto":
        url = f"https://api.polygon.io/v3/reference/tickers/X:{symbol}"
    else:  # Forex
        url = f"https://api.polygon.io/v3/reference/tickers/C:{symbol}"
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(url, headers=headers)
    
    return response.status_code == 200

def get_trading_days(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
    return trading_days.tz_localize('US/Eastern')

def get_trading_days_lookback(days_lookback):
    nyse = mcal.get_calendar('NYSE')
    end_date = pd.Timestamp.now(tz='US/Eastern')
    start_date = (end_date - pd.Timedelta(days=days_lookback)).tz_convert('US/Eastern')
    trading_days = nyse.valid_days(start_date=start_date.tz_localize(None), end_date=end_date.tz_localize(None))
    return trading_days.tz_convert('US/Eastern')

def fetch_stock_data(ticker, start_date, end_date, interval="1/minute"):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{interval}/{start_date}/{end_date}"
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

def fetch_forex_data(pair, start_date, end_date, interval="1/minute"):
    url = f"https://api.polygon.io/v2/aggs/ticker/C:{pair}/range/{interval}/{start_date}/{end_date}"
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

def fetch_crypto_data(pair, start_date, end_date, interval="1/minute"):
    url = f"https://api.polygon.io/v2/aggs/ticker/X:{pair}/range/{interval}/{start_date}/{end_date}"
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
            st.warning(f"No data available for Crypto Pair {pair} in the given date range.")
            return pd.DataFrame()
    else:
        st.error(f"Error fetching crypto data for {pair}: {response.text}")
        return pd.DataFrame()

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
        elif data_type == "Crypto":
            df = fetch_crypto_data(symbol, start_date, end_date)
        else:  # Forex
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

def fetch_stock_data_in_chunks(api_key, ticker, start_date, end_date, interval, chunk_size=52):
    """Fetch stock data in chunks for a given ticker."""
    nyse = mcal.get_calendar('NYSE')
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
    
    df_all = pd.DataFrame()
    
    for i in range(0, len(trading_days), chunk_size):
        start_date_range = trading_days[i].date().isoformat()
        end_date_range = trading_days[min(i + chunk_size, len(trading_days) - 1)].date().isoformat()
        
        # Use the existing fetch_stock_data function with the interval parameter
        interval_mapping = {
            "1 minute": "1/minute",
            "5 minutes": "5/minute",
            "1 hour": "1/hour",
            "1 day": "1/day"
        }
        api_interval = interval_mapping.get(interval)
        
        df_chunk = fetch_stock_data(ticker, start_date_range, end_date_range, api_interval)
        if not df_chunk.empty:
            df_all = pd.concat([df_all, df_chunk])
        
        time.sleep(12)  # Rate limiting

    if not df_all.empty:
        # Create separate columns for each OHLCV value
        renamed_cols = {}
        for col in df_all.columns:
            if col == 'o':
                renamed_cols[col] = f"{ticker}_open"
            elif col == 'h':
                renamed_cols[col] = f"{ticker}_high"
            elif col == 'l':
                renamed_cols[col] = f"{ticker}_low"
            elif col == 'c':
                renamed_cols[col] = f"{ticker}_close"
            elif col == 'v':
                renamed_cols[col] = f"{ticker}_volume"
            elif col == 'vw':
                renamed_cols[col] = f"{ticker}_vwap"
            elif col == 'n':
                renamed_cols[col] = f"{ticker}_trades"
            
        df_all = df_all.rename(columns=renamed_cols)

    return df_all

def fetch_forex_data_in_chunks(api_key, pair, start_date, end_date, interval, chunk_size=30):
    """Fetch forex data in chunks for a given pair."""
    nyse = mcal.get_calendar('NYSE')
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
    
    df_all = pd.DataFrame()
    
    for i in range(0, len(trading_days), chunk_size):
        start_date_range = trading_days[i].date().isoformat()
        end_date_range = trading_days[min(i + chunk_size, len(trading_days) - 1)].date().isoformat()
        
        interval_mapping = {
            "1 minute": "1/minute",
            "5 minutes": "5/minute",
            "1 hour": "1/hour",
            "1 day": "1/day"
        }
        api_interval = interval_mapping.get(interval)
        
        df_chunk = fetch_forex_data(pair, start_date_range, end_date_range, api_interval)
        if not df_chunk.empty:
            # Create separate columns for each OHLCV value
            renamed_cols = {}
            for col in df_chunk.columns:
                if col == 'o':
                    renamed_cols[col] = f"{pair}_open"
                elif col == 'h':
                    renamed_cols[col] = f"{pair}_high"
                elif col == 'l':
                    renamed_cols[col] = f"{pair}_low"
                elif col == 'c':
                    renamed_cols[col] = f"{pair}_close"
                elif col == 'v':
                    renamed_cols[col] = f"{pair}_volume"
                elif col == 'vw':
                    renamed_cols[col] = f"{pair}_vwap"
                elif col == 'n':
                    renamed_cols[col] = f"{pair}_trades"
            
            df_chunk = df_chunk.rename(columns=renamed_cols)
            df_all = pd.concat([df_all, df_chunk])
        
        time.sleep(12)  # Rate limiting

    return df_all

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

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.title = None
    st.session_state.symbol = None
    st.session_state.data_type = None

# Streamlit UI
st.title("Financial Data Explorer")

# Mode selection
mode = st.radio("Choose Mode:", ("Stock Analysis", "Forex Analysis", "Crypto Analysis", "Multi-Ticker Download"))

if mode == "Multi-Ticker Download":
    # Data type selection
    data_type = st.radio("Choose Data Type:", ("Stock", "Forex", "Crypto"))
    
    # Ticker input
    ticker_input = st.text_area(
        "Enter Tickers (comma-separated):",
        help="Enter multiple tickers (e.g., EURUSD, EURGBP for forex or AAPL, MSFT for stocks)"
    )
    
    # Parse tickers
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Interval selection
    interval = st.selectbox(
        "Select Interval:",
        ["1 minute", "5 minutes", "1 hour", "1 day"]
    )
    
    if st.button("Download Data"):
        if not tickers:
            st.error("Please enter at least one ticker")
        else:
            # Verify tickers
            invalid_tickers = []
            with st.spinner('Verifying tickers...'):
                for ticker in tickers:
                    if not verify_ticker(ticker, data_type):
                        invalid_tickers.append(ticker)
            
            if invalid_tickers:
                st.error(f"Invalid tickers: {', '.join(invalid_tickers)}")
            else:
                all_data = []
                progress_bar = st.progress(0)
                
                for i, ticker in enumerate(tickers):
                    st.text(f"Fetching data for {ticker}...")
                    if data_type == "Forex":
                        df = fetch_forex_data_in_chunks(API_KEY, ticker, start_date.isoformat(), end_date.isoformat(), interval)
                    elif data_type == "Crypto":
                        interval_mapping = {
                            "1 minute": "1/minute",
                            "5 minutes": "5/minute",
                            "1 hour": "1/hour",
                            "1 day": "1/day"
                        }
                        api_interval = interval_mapping.get(interval)
                        df = fetch_crypto_data(ticker, start_date.isoformat(), end_date.isoformat(), api_interval)
                    else:
                        df = fetch_stock_data_in_chunks(API_KEY, ticker, start_date.isoformat(), end_date.isoformat(), interval)
                    
                    if not df.empty:
                        # Create separate columns for each OHLCV value
                        renamed_cols = {}
                        for col in df.columns:
                            if col == 'o':
                                renamed_cols[col] = f"{ticker}_open"
                            elif col == 'h':
                                renamed_cols[col] = f"{ticker}_high"
                            elif col == 'l':
                                renamed_cols[col] = f"{ticker}_low"
                            elif col == 'c':
                                renamed_cols[col] = f"{ticker}_close"
                            elif col == 'v':
                                renamed_cols[col] = f"{ticker}_volume"
                            elif col == 'vw':
                                renamed_cols[col] = f"{ticker}_vwap"
                            elif col == 'n':
                                renamed_cols[col] = f"{ticker}_trades"
                        
                        df = df.rename(columns=renamed_cols)
                        
                        # Ensure index is unique before adding to all_data
                        if not df.index.is_unique:
                            df = df[~df.index.duplicated(keep='last')]
                        
                        all_data.append(df)
                    
                    progress_bar.progress((i + 1) / len(tickers))
                    time.sleep(12)  # Rate limiting
                
                progress_bar.empty()
                
                if all_data:
                    try:
                        # Merge all dataframes
                        final_df = pd.concat(all_data, axis=1)
                        
                        # Remove any duplicate columns
                        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
                        
                        # Sort index to ensure chronological order
                        final_df = final_df.sort_index()
                        
                        st.success("Data fetched successfully!")
                        
                        # Display data info
                        st.subheader("Data Summary")
                        st.write(f"Date Range: {final_df.index.min()} to {final_df.index.max()}")
                        st.write(f"Total Records: {len(final_df)}")
                        
                        # Display preview
                        st.subheader("Preview of Combined Data")
                        st.dataframe(final_df)
                        
                        # Add column information
                        st.subheader("Column Information")
                        st.markdown("""
                        For each ticker, the following columns are created:
                        - _open: Opening price
                        - _high: Highest price
                        - _low: Lowest price
                        - _close: Closing price
                        - _volume: Trading volume (if available)
                        - _vwap: Volume-weighted average price (if available)
                        - _trades: Number of trades (if available)
                        """)
                        
                        # Download button
                        csv = final_df.to_csv()
                        st.download_button(
                            label="Download Combined Data as CSV",
                            data=csv,
                            file_name=f"multi_ticker_data_{start_date}_{end_date}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error merging data: {str(e)}")
                        st.error("Please try again with a smaller date range or fewer tickers")
                else:
                    st.error("No data available for the selected tickers and time range")
 # Single Ticker Analysis or
else:  # Single Ticker Analysis or Forex Analysis
    # Data type selection
    data_type = "Crypto" if mode == "Crypto Analysis" else "Forex" if mode == "Forex Analysis" else "Stock"

    # Manual symbol input with help text
    if data_type == "Stock":
        help_text = "Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
    elif data_type == "Crypto":
        help_text = "Enter crypto pairs (e.g., BTCUSD, ETHUSD)"
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
                    trading_days = get_trading_days_lookback(days_lookback)
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
