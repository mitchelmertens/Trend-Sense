# TrendSense

## Overview

The **Stock Sentiment Analyzer** is a Python script that uses sentiment analysis on recent news articles related to a specified stock symbol. By analyzing the sentiment of news articles and comparing it with historical stock price data, the script provides valuable insights into potential market trends. The project uses the VADER sentiment analysis tool, NewsAPI for fetching news articles, and Yahoo Finance (yfinance) for obtaining stock price data.

## Features

- **Sentiment Analysis of News Articles**: Retrieves recent news articles about a specified stock and performs sentiment analysis to determine whether the sentiment is positive, negative, or neutral.
- **Historical Stock Price Data**: Fetches and analyzes historical stock price data to observe trends and correlations with news sentiment.
- **Trend and Correlation Analysis**: Calculates the correlation between sentiment scores and stock prices and performs trend analysis using linear regression.
- **Top Positive and Negative Articles**: Displays the top 3 most positive and negative articles to provide context for the sentiment analysis.
- **Recommendation System**: Provides stock trading recommendations (Buy, Sell, or Hold) based on the trend analysis and sentiment correlation.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages: `yfinance`, `newsapi-python`, `nltk`, `matplotlib`, `pandas`, `scipy`
- NewsAPI account and API key (You can obtain one by signing up at [NewsAPI](https://newsapi.org/))

## Acknowledgments

- [NewsAPI](https://newsapi.org/) for providing access to news articles.
- [Yahoo Finance](https://finance.yahoo.com/) for stock price data.
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) for sentiment scoring.

