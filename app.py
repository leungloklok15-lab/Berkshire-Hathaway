import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# App configuration
st.set_page_config(
    page_title="Berkshire Hathaway 13F Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import yfinance with fallback
try:
    import yfinance as yf
except ImportError:
    st.error("""
    **yfinance package is missing!**
    
    Please make sure you've included it in your requirements.txt file:
    ```
    yfinance==0.2.36
    ```
    """)
    st.stop()

# Load sample 13F data
@st.cache_data
def load_13f_data():
    portfolio = {
        'Stock': ['AAPL', 'BAC', 'CVX', 'KO', 'AXP', 'KHC', 'OXY', 'MCO', 'TSM', 'ATVI'],
        'Shares': [915560000, 1010000000, 110247000, 400000000, 151610000, 325634000, 211666000, 24693000, 60000000, 52700000],
        'Value ($B)': [155.6, 33.4, 29.2, 23.6, 22.4, 11.9, 11.1, 9.9, 6.1, 3.9]
    }
    return pd.DataFrame(portfolio)

# Fetch stock data
@st.cache_data
def get_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Calculate technical indicators
def calculate_indicators(df, ma_periods=[20, 50], bb_period=20, rsi_period=14):
    if df.empty:
        return df
        
    try:
        # Moving Averages
        for period in ma_periods:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # Bollinger Bands
        df['BB_MA'] = df['Close'].rolling(window=bb_period).mean()
        df['BB_upper'] = df['BB_MA'] + 2 * df['Close'].rolling(window=bb_period).std()
        df['BB_lower'] = df['BB_MA'] - 2 * df['Close'].rolling(window=bb_period).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        # Handle division by zero
        rs = np.zeros_like(avg_gain)
        valid = avg_loss > 0
        rs[valid] = avg_gain[valid] / avg_loss[valid]
        
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Fill NaN with neutral value
        
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
    
    return df

# Plot candlestick chart with indicators
def plot_candlestick(df, title, show_ma=True, show_bb=True, show_rsi=True):
    if df.empty:
        return go.Figure()
        
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Moving Averages
    if show_ma:
        for col in df.columns:
            if 'MA_' in col:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    line=dict(width=1.5)
                ))
    
    # Bollinger Bands
    if show_bb:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_upper'],
            name='BB Upper',
            line=dict(color='rgba(200, 200, 200, 0.5)')
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_lower'],
            name='BB Lower',
            fill='tonexty',
            line=dict(color='rgba(200, 200, 200, 0.5)')
        ))
    
    # Layout settings
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template=st.session_state.get('chart_style', 'plotly_white'),
        height=500,
        showlegend=True
    )
    
    # Create RSI subplot if enabled
    if show_rsi and 'RSI' in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=1.5)
        ))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.update_layout(
            height=200,
            showlegend=False,
            template=st.session_state.get('chart_style', 'plotly_white')
        )
        
        fig = go.Figure(fig)  # Convert to Figure object
        fig.add_trace(fig_rsi.data[0], row=2, col=1)
        fig.update_layout(grid={'rows': 2, 'columns': 1, 'pattern': "independent"})
        fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    return fig

# Calculate financial ratios
def get_financial_ratios(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        ratios = {
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'P/B Ratio': info.get('priceToBook', 'N/A'),
            'ROE': info.get('returnOnEquity', 'N/A'),
            'Current Ratio': info.get('currentRatio', 'N/A'),
            'Debt/Equity': info.get('debtToEquity', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') else 'N/A'
        }
        
        return ratios
    except:
        return {
            'P/E Ratio': 'N/A',
            'P/B Ratio': 'N/A',
            'ROE': 'N/A',
            'Current Ratio': 'N/A',
            'Debt/Equity': 'N/A',
            'Dividend Yield': 'N/A'
        }

# Correlation analysis
def plot_correlation(portfolio, period="1y"):
    tickers = portfolio['Stock'].tolist()
    data = {}
    
    for ticker in tickers:
        try:
            df = get_stock_data(ticker, period)
            if not df.empty:
                data[ticker] = df['Close'].pct_change().dropna()
        except:
            continue
    
    if not data or len(data) < 2:
        return None
    
    corr_df = pd.DataFrame(data)
    corr_matrix = corr_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale=st.session_state.get('correlation_colormap', 'RdBu'),
        zmin=-1,
        zmax=1,
        title="Portfolio Correlation Matrix"
    )
    fig.update_layout(height=600)
    return fig

# Initialize session state
if 'chart_config' not in st.session_state:
    st.session_state.chart_config = {
        'ma_periods': [20, 50],
        'bb_period': 20,
        'rsi_period': 14,
        'chart_style': 'plotly_white',
        'correlation_colormap': 'RdBu'
    }

# App header
st.title("📈 Berkshire Hathaway 13F Analyzer")
st.markdown("""
**Track Warren Buffett's investment portfolio** with real-time data, technical indicators, and financial analysis.
""")

# Load 13F data
portfolio = load_13f_data()

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Select Timeframe", 
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "Max"],
        index=3
    )
    
    # Technical indicators
    st.subheader("Technical Indicators")
    show_ma = st.checkbox("Moving Averages", True)
    show_bb = st.checkbox("Bollinger Bands", True)
    show_rsi = st.checkbox("RSI", True)
    
    # Customization
    st.subheader("Chart Customization")
    chart_style = st.selectbox(
        "Chart Style", 
        ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
        index=1
    )
    st.session_state.chart_style = chart_style
    
    correlation_colormap = st.selectbox(
        "Correlation Colors", 
        ["RdBu", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
        index=0
    )
    st.session_state.correlation_colormap = correlation_colormap
    
    # Portfolio upload
    st.subheader("Portfolio Analysis")
    uploaded_file = st.file_uploader(
        "Upload Your Portfolio (CSV)", 
        type=["csv"],
        help="Upload a CSV with columns: 'Stock', 'Shares', 'Value'"
    )
    
    if uploaded_file:
        try:
            user_portfolio = pd.read_csv(uploaded_file)
            st.success("Portfolio uploaded successfully!")
            portfolio = user_portfolio
        except:
            st.error("Invalid file format. Please use the required format.")

# Portfolio overview
st.header("Portfolio Holdings")
st.dataframe(portfolio.sort_values('Value ($B)', ascending=False).reset_index(drop=True))

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Stock Analysis", 
    "Technical Charts", 
    "Financial Ratios",
    "Correlation Analysis"
])

# Stock Analysis Tab
with tab1:
    st.subheader("Individual Stock Analysis")
    selected_stock = st.selectbox("Select Stock", portfolio['Stock'])
    
    if selected_stock:
        try:
            stock_data = get_stock_data(selected_stock, timeframe)
            
            if stock_data.empty:
                st.warning(f"No data available for {selected_stock}")
            else:
                # Calculate daily returns
                stock_data['Daily Return'] = stock_data['Close'].pct_change()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
                    st.metric("52-Week High", f"${stock_data['High'].max():.2f}")
                
                with col2:
                    change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]
                    pct_change = (change / stock_data['Close'].iloc[0]) * 100
                    st.metric(f"Performance ({timeframe})", f"${change:.2f}", f"{pct_change:.2f}%")
                    st.metric("52-Week Low", f"${stock_data['Low'].min():.2f}")
                
                # Price history chart
                st.subheader(f"{selected_stock} Price History")
                fig = px.line(stock_data, x=stock_data.index, y='Close')
                fig.update_layout(
                    template=st.session_state.chart_style,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data for {selected_stock}: {str(e)}")

# Technical Charts Tab
with tab2:
    st.subheader("Technical Analysis")
    
    if selected_stock:
        try:
            stock_data = get_stock_data(selected_stock, timeframe)
            
            if stock_data.empty:
                st.warning(f"No data available for {selected_stock}")
            else:
                stock_data = calculate_indicators(stock_data)
                
                # Chart configuration
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"{selected_stock} Technical Chart")
                    fig = plot_candlestick(
                        stock_data, 
                        f"{selected_stock} Analysis",
                        show_ma,
                        show_bb,
                        show_rsi
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export options
                    export_col1, export_col2 = st.columns(2)
                    with export_col1:
                        st.download_button(
                            label="Download Chart as HTML",
                            data=fig.to_html(),
                            file_name=f"{selected_stock}_chart.html",
                            mime="text/html"
                        )
                    with export_col2:
                        st.download_button(
                            label="Download Chart as PNG",
                            data=fig.to_image(format="png"),
                            file_name=f"{selected_stock}_chart.png",
                            mime="image/png"
                        )
                
                with col2:
                    st.subheader("Chart Settings")
                    rsi_threshold = st.slider(
                        "RSI Overbought/Oversold Thresholds",
                        50, 90, (30, 70))
                    st.info(f"RSI > {rsi_threshold[1]} = Overbought")
                    st.info(f"RSI < {rsi_threshold[0]} = Oversold")
        except Exception as e:
            st.error(f"Error generating technical chart: {str(e)}")

# Financial Ratios Tab
with tab3:
    st.subheader("Financial Health Analysis")
    
    if selected_stock:
        ratios = get_financial_ratios(selected_stock)
        
        col1, col2, col3 = st.columns(3)
        
        # Valuation metrics
        with col1:
            st.subheader("Valuation")
            st.metric("P/E Ratio", f"{ratios['P/E Ratio']:.2f}" if isinstance(ratios['P/E Ratio'], float) else "N/A")
            st.metric("P/B Ratio", f"{ratios['P/B Ratio']:.2f}" if isinstance(ratios['P/B Ratio'], float) else "N/A")
            st.metric("Dividend Yield", f"{ratios['Dividend Yield']:.2f}%" if isinstance(ratios['Dividend Yield'], float) else "N/A")
        
        # Profitability metrics
        with col2:
            st.subheader("Profitability")
            st.metric("ROE", f"{ratios['ROE']:.2%}" if isinstance(ratios['ROE'], float) else "N/A")
            st.metric("Operating Margin", "N/A")
        
        # Financial health metrics
        with col3:
            st.subheader("Financial Health")
            st.metric("Current Ratio", f"{ratios['Current Ratio']:.2f}" if isinstance(ratios['Current Ratio'], float) else "N/A")
            st.metric("Debt/Equity", f"{ratios['Debt/Equity']:.2f}" if isinstance(ratios['Debt/Equity'], float) else "N/A")
        
        # Benchmark comparison
        st.subheader("Sector Comparison")
        st.warning("Sector comparison data would be implemented with a market data API in production")

# Correlation Analysis Tab
with tab4:
    st.subheader("Portfolio Correlation Analysis")
    
    if len(portfolio) > 1:
        with st.expander("Correlation Matrix Explanation"):
            st.markdown("""
            **How to interpret this matrix:**
            - Values close to 1 (dark blue) indicate strong positive correlation
            - Values close to -1 (dark red) indicate strong negative correlation
            - Values near 0 indicate no correlation
            
            A well-diversified portfolio should contain assets with varying correlations.
            """)
        
        try:
            fig = plot_correlation(portfolio, timeframe)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data to calculate correlations")
        except Exception as e:
            st.error(f"Error generating correlation matrix: {str(e)}")
    else:
        st.warning("Add at least 2 stocks to portfolio to see correlation analysis")

# Footer
st.markdown("---")
st.markdown("""
**Data Sources:**  
- Historical Price Data: Yahoo Finance (yfinance)  
- 13F Holdings: SEC EDGAR Database (sample data)  

**Note:** This app uses sample Berkshire Hathaway portfolio data.  
For production use, implement SEC EDGAR API integration for live 13F data.
""")
