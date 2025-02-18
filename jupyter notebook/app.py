import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import datetime, timedelta
import os
from database import Session, User, Stock, StockData, Prediction
from sqlalchemy import func

# Configure page
st.set_page_config(page_title="StockSage", layout="wide")

# Initialize database session
session = Session()

# Custom CSS for styling
st.markdown("""
    <style>
        .metric-card {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin: 10px;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("StockSage")
st.subheader("AI-powered Stock Analysis Tool")

# Sidebar for user management
with st.sidebar:
    st.header("Your Profile")
    username = st.text_input("Your Name")
    email = st.text_input("Email")
    
    if st.button("Save Profile"):
        if username and email:
            user = session.query(User).filter_by(username=username).first()
            if not user:
                new_user = User(
                    username=username,
                    email=email,
                    created_at=datetime.now().date()
                )
                session.add(new_user)
                session.commit()
                st.success(f"Welcome, {username}!")
            else:
                st.success(f"Welcome back, {username}!")
        else:
            st.warning("Please enter both name and email")

# Stock Ticker Input
user_input = st.text_input('Enter Stock Ticker', '')
submit_button = st.button("Analyze Stock")

if submit_button and user_input:
    try:
        # Define time period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5 years of data
        
        # Check if stock exists in database
        stock = session.query(Stock).filter_by(ticker=user_input.upper()).first()
        if not stock:
            # Create new stock entry
            stock = Stock(
                ticker=user_input.upper(),
                company_name=yf.Ticker(user_input).info.get('longName', 'Unknown'),
                sector=yf.Ticker(user_input).info.get('sector', 'Unknown'),
                industry=yf.Ticker(user_input).info.get('industry', 'Unknown')
            )
            session.add(stock)
            session.commit()
        
        # Download stock data with error handling
        @st.cache_data(ttl=3600)  # Cache data for 1 hour
        def load_stock_data(ticker, start, end):
            try:
                data = yf.download(ticker, start=start, end=end)
                if data.empty:
                    raise ValueError("No data found for this ticker")
                return data
            except Exception as e:
                raise Exception(f"Error downloading data: {str(e)}")
        
        # Load data with progress indicator
        with st.spinner('Downloading stock data...'):
            df = load_stock_data(user_input, start_date, end_date)
            
            # Store stock data in database
            for index, row in df.iterrows():
                stock_data = StockData(
                    stock_id=stock.stock_id,
                    date=index.date(),
                    open_price=float(row['Open']),
                    high_price=float(row['High']),
                    low_price=float(row['Low']),
                    close_price=float(row['Close']),
                    volume=int(row['Volume'])
                )
                session.merge(stock_data)
            session.commit()
        
        # Display stock details and metrics
        st.markdown(f"## {user_input.upper()} Stock Analysis")
        st.markdown(f"**Company:** {stock.company_name}")
        st.markdown(f"**Sector:** {stock.sector} | **Industry:** {stock.industry}")
        
        # Calculate key metrics safely
        try:
            current_price = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            price_change = ((current_price - prev_close) / prev_close) * 100
            high_52week = float(df['High'].tail(252).max())
            low_52week = float(df['Low'].tail(252).min())
            avg_volume = int(df['Volume'].tail(30).mean())
            volatility = float(df['Close'].pct_change().std() * np.sqrt(252) * 100)
            
            # Format metrics
            metrics = {
                "Current Price": f"${current_price:.2f}",
                "Daily Change": f"{price_change:+.2f}%",
                "52-Week High": f"${high_52week:.2f}",
                "52-Week Low": f"${low_52week:.2f}",
                "30-Day Avg Volume": f"{avg_volume:,}",
                "Annual Volatility": f"{volatility:.1f}%"
            }
            
            # Display metrics in columns
            cols = st.columns(3)
            for i, (metric, value) in enumerate(metrics.items()):
                with cols[i % 3]:
                    st.metric(
                        label=metric,
                        value=value,
                        delta=price_change if metric == "Current Price" else None
                    )
        
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            
        # Create tabs for analysis
        tab1, tab2, tab3, tab4 = st.tabs(["Historical Data", "Technical Analysis", "Price Predictions", "Community Predictions"])
        
        with tab1:
            try:
                st.subheader("Historical Data Analysis")
                
                # Date range selector
                date_range = st.date_input(
                    "Select Date Range",
                    value=(df.index.min().date(), df.index.max().date()),
                    min_value=df.index.min().date(),
                    max_value=df.index.max().date()
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
                    filtered_df = df[mask]
                    
                    # Display statistics
                    st.write("Statistical Summary")
                    st.dataframe(filtered_df.describe())
                    
                    # Display raw data with pagination
                    st.write("Raw Data")
                    st.dataframe(filtered_df)
                    
            except Exception as e:
                st.error(f"Error in historical analysis: {str(e)}")
        
        with tab2:
            try:
                st.subheader("Technical Analysis")
                
                # Price and volume chart
                fig = plt.figure(figsize=(12, 8))
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                # Plot price
                ax1.plot(df.index, df['Close'], color='blue', label='Price')
                
                # Plot volume as line instead of bars for better performance
                ax2.plot(df.index, df['Volume'], color='gray', alpha=0.3, label='Volume')
                
                ax1.set_ylabel('Price ($)', color='blue')
                ax2.set_ylabel('Volume', color='gray')
                ax1.grid(True, alpha=0.3)
                
                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                plt.title('Price and Volume Over Time')
                st.pyplot(fig)
                plt.close(fig)
                
                # Moving averages
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df.index, df['Close'], label='Price', color='blue')
                ax.plot(df.index, df['Close'].rolling(window=100).mean(), label='100-day MA', color='red')
                ax.plot(df.index, df['Close'].rolling(window=200).mean(), label='200-day MA', color='green')
                ax.set_ylabel('Price ($)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.title('Moving Averages')
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Error in technical analysis: {str(e)}")
        
        with tab3:
            try:
                st.subheader("Price Predictions")
                
                # Calculate exponential moving averages
                df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                
                # Create future dates
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date, periods=30, freq='B')
                
                # Calculate trend using recent EMAs
                recent_trend = (df['EMA_12'].iloc[-1] - df['EMA_12'].iloc[-20]) / df['EMA_12'].iloc[-20]
                
                # Generate predictions
                last_price = df['Close'].iloc[-1]
                predictions = [last_price]
                
                for _ in range(len(future_dates)-1):
                    next_price = predictions[-1] * (1 + recent_trend)
                    predictions.append(next_price)
                
                # Create the plot
                fig = plt.figure(figsize=(12, 6))
                
                # Plot historical data and EMAs
                plt.plot(df.index[-60:], df['Close'][-60:], label='Historical Price')
                plt.plot(df.index[-60:], df['EMA_12'][-60:], label='12-day EMA', alpha=0.7)
                plt.plot(df.index[-60:], df['EMA_26'][-60:], label='26-day EMA', alpha=0.7)
                
                # Plot prediction
                plt.plot(future_dates, predictions, 'r--', label='Predicted Price')
                
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.title('Price Prediction with Exponential Moving Averages')
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Show predicted values
                prediction_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': predictions
                })
                st.dataframe(prediction_df.set_index('Date'))

                 # Store prediction in database if user is logged in
                if username:
                    user = session.query(User).filter_by(username=username).first()
                    if user:
                        prediction = Prediction(
                            stock_id=stock.stock_id,
                            user_id=user.user_id,
                            prediction_date=datetime.now().date(),
                            predicted_price=predictions[-1],
                            prediction_for_date=future_dates[-1].date()
                        )
                        session.add(prediction)
                        session.commit()
                        st.success("Prediction saved to database!")
                
            except Exception as e:
                st.error(f"Error in price prediction: {str(e)}")
                st.write("Detailed error:", str(e))

        with tab4:
            st.subheader("Community Predictions")
            
            # Display recent community predictions
            recent_predictions = (
            session.query(
                User.username,
                Prediction.predicted_price,
                Prediction.prediction_for_date
            )
            .join(User)
            .filter(Prediction.stock_id == stock.stock_id)
            .order_by(Prediction.prediction_date.desc())
            .limit(5)  # Only show last 5 predictions
            .all()
            )
            
            if recent_predictions:
                st.subheader("Recent Community Predictions")
                for pred in recent_predictions:
                    st.write(f"{pred.username}: ${pred.predicted_price:.2f} (for {pred.prediction_for_date})")

            else:
                st.info("No community predictions available yet.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check the ticker symbol and try again")
    
    session.close()

else:
    st.info("Enter a stock ticker symbol and click 'Analyze Stock' to begin")