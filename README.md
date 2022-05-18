# twusk
A Data Science project which explores the relationship between Elon Musk's tweets and the TSLA stock prices.

Includes
<\b>Data:
        -tweets: raw Tweet data from https://www.kaggle.com/datasets/ayhmrba/elon-musk-tweets-2010-2021 (retreived March 25th 2022)
        -needed:
            -raw: raw stock price data, some copies of raw Tweet data
                TSLA_2017_2022: https://www.nasdaq.com/market-activity/stocks/tsla/historical (retreived March 25th 2022)
                TSLA_2010_2018: https://finance.yahoo.com/quote/TSLA/history?period1=1277769600&period2=1535414400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true (retreived March 31th 2022, replacement for TSLA_2010_2019)
            -cleaned: cleaned data sets
                Stock: All stock data sets combined
                Tweets: All Tweet data sets combined
                Tweets_and_Stock: Outer merge of tweet and stock data sets by day
                word_count: number of times a word appears in all Tweets
    EDA:
        -EDA notebook
        -Interactive data explorer (uses Bokeh, to use this, it is easiest to run "start_interactive_explorer.py" which will start it.)
    Other files:
        -ARIMA: ARIMA models of stock price per week and month
        -count_words.py: Script to count occurences of words in the Tweets data set and generate "word_count.csv"
        -data_cleaning.ipynb: Jupyter notebook to generate the cleaned data sets out of the raw data
        -data_cleaning.py: Python script version of "data_cleaning.ipynb" notebook
        -neural_net.ipynb: Uses a neural net to try predicting stock price based on number of tweets
