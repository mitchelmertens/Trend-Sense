import yfinance as yf
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import logging
from scipy.stats import linregress

# Ensure you have the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize the News API client
newsapi = NewsApiClient(api_key='YOUR-NEWS-API-KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)

def classify_sentiment(score):
    """
    Classifies sentiment score into Positive, Negative, or Neutral.
    """
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

def get_articles_sentiments(stock_symbol, start_date, end_date):
    """
    Fetches news articles about the specified stock and performs sentiment analysis.
    """
    articles_list = []
    current_date = start_date

    while current_date <= end_date:
        try:
            articles = newsapi.get_everything(
                q=stock_symbol,
                from_param=current_date.isoformat(),
                to=(current_date + timedelta(days=1)).isoformat(),
                language="en",
                sort_by="relevancy",
                page_size=100
            )
            for article in articles['articles']:
                content = f"{article['title']}. {article['description']}"
                sentiment = sia.polarity_scores(content)['compound']
                articles_list.append({
                    'date': current_date,
                    'sentiment': sentiment,
                    'url': article['url'],
                    'title': article['title'],
                    'description': article['description']
                })
        except Exception as e:
            logging.error(f"Error fetching articles for {current_date}: {e}")

        current_date += timedelta(days=1)

    return pd.DataFrame(articles_list)

def get_stock_data(stock_symbol, start_date, end_date):
    """
    Fetches historical stock data for the specified stock symbol.
    """
    try:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        return stock_data[['Close']]
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def calculate_trend_recommendation(sentiment_vs_stock):
    """
    Calculates the correlation and provides a recommendation based on trend analysis.
    """
    # Drop NaN values for correlation calculation
    sentiment_vs_stock = sentiment_vs_stock.dropna()

    if sentiment_vs_stock.empty:
        return None, None, "Not enough data"

    # Calculate correlation
    correlation = sentiment_vs_stock['Close'].corr(sentiment_vs_stock['sentiment'])

    # Perform trend analysis using linear regression
    trend = linregress(sentiment_vs_stock.index.to_julian_date(), sentiment_vs_stock['Close'])
    slope = trend.slope

    # Provide recommendation based on trend slope
    recommendation = "Hold"
    if slope > 0:
        recommendation = "Buy"
    elif slope < 0:
        recommendation = "Sell"

    return correlation, slope, recommendation

# User input for stock symbol
stock_symbol = input("Enter the stock symbol (e.g., AAPL for Apple): ").upper()

# Define the date range
end_date = date.today()
start_date = end_date - timedelta(days=30)

# Fetch news sentiment data
sentiment_df = get_articles_sentiments(stock_symbol, start_date, end_date)

# Fetch historical stock data
stock_data = get_stock_data(stock_symbol, start_date, end_date)

# Check if data was fetched successfully
if sentiment_df.empty or stock_data.empty:
    logging.error("No data available for the specified dates.")
else:
    # Aggregate daily sentiment
    daily_sentiment = sentiment_df.groupby('date')['sentiment'].mean()
    sentiment_vs_stock = stock_data.join(daily_sentiment, how='left')

    # Calculate correlation and recommendation
    correlation, trend_slope, recommendation = calculate_trend_recommendation(sentiment_vs_stock)

    # Print summary of sentiments
    sentiment_counts = sentiment_df['sentiment'].apply(classify_sentiment).value_counts()
    print("\nSentiment Summary:")
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment}: {count} articles")

    # Top 3 most positive and negative articles
    print("\nTop 3 Positive Articles:")
    top_positive = sentiment_df.nlargest(3, 'sentiment')
    for idx, article in top_positive.iterrows():
        print(f"- {article['title']} ({article['sentiment']:.2f})")
        print(f"  URL: {article['url']}")

    print("\nTop 3 Negative Articles:")
    top_negative = sentiment_df.nsmallest(3, 'sentiment')
    for idx, article in top_negative.iterrows():
        print(f"- {article['title']} ({article['sentiment']:.2f})")
        print(f"  URL: {article['url']}")

    # Print correlation and recommendation
    print(f"\nCorrelation between sentiment and stock price: {correlation:.2f}")
    print(f"Trend slope: {trend_slope:.2f}")
    print(f"Recommendation: {recommendation}")

    # Plotting the results
    plt.figure(figsize=(14, 8))

    # Plot stock price
    plt.subplot(2, 1, 1)
    plt.plot(sentiment_vs_stock.index, sentiment_vs_stock['Close'], label='Stock Price', color='blue')
    plt.title(f'{stock_symbol} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()

    # Add trend line for stock price
    x_dates = sentiment_vs_stock.index.to_julian_date()
    slope, intercept = np.polyfit(x_dates, sentiment_vs_stock['Close'], 1)
    plt.plot(sentiment_vs_stock.index, intercept + slope * x_dates, '--', color='blue', alpha=0.5)
    plt.text(sentiment_vs_stock.index[-1], sentiment_vs_stock['Close'].max(), f'Trend slope: {trend_slope:.2f}', color='blue')

    # Plot sentiment
    plt.subplot(2, 1, 2)
    plt.plot(sentiment_vs_stock.index, sentiment_vs_stock['sentiment'], label='Sentiment', color='orange')
    plt.title(f'{stock_symbol} News Sentiment')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.grid(True)
    plt.legend()

    # Add correlation text
    plt.text(sentiment_vs_stock.index[-1], sentiment_vs_stock['sentiment'].min(), f'Correlation: {correlation:.2f}', color='orange')

    plt.tight_layout()
    plt.show()
