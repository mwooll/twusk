# data_preparation

import numpy as np
import pandas as pd

""" Preparing the Data """
def make_DataFrames(tweet_columns, starting_date, ending_date):
    """ Reading in the data """
    # Reading the wanted columns of the tweets data into a dataframe.
    tweets_file_name = "../data/needed/cleaned/Tweets.csv"

    tweets_df = pd.read_csv(tweets_file_name, index_col=None,
                            usecols=tweet_columns)

    tweets_df["date"] = pd.to_datetime(tweets_df["date"]).dt.date

    # Reading the stock data into a dataframe.
    stocks_file_name = "../data/needed/cleaned/Stock.csv"
    stocks_df = pd.read_csv(stocks_file_name, index_col=None)
    stocks_df["date"] = pd.to_datetime(stocks_df["date"])
    stocks_df.set_index("date", inplace=True,
                        verify_integrity=False, drop=True)
    stocks_df["close - open"] = stocks_df["close"] - stocks_df["open"]
    stocks_df.index.names = ["index"]
    stocks_df["date"] = stocks_df.index
    stocks_df.sort_index(inplace=True, ascending=True)
    # print(stocks_df)


    # extracting the date range
    if starting_date == None:
        tweet_min, stock_min = min(tweets_df["date"]), min(stocks_df.index)
        starting_date = min(pd.Timestamp(tweet_min), pd.Timestamp(stock_min))
    if ending_date == None:
        tweet_max, stock_max = max(tweets_df["date"]), max(stocks_df.index)
        ending_date = min(pd.Timestamp(tweet_max), pd.Timestamp(stock_max))

    date_range = pd.date_range(start=starting_date, end=ending_date, freq="D")

    # Counting how many tweets there are for each day
    counts = tweets_df["date"].value_counts()
    counts.sort_index(inplace=True)
    tweet_count = pd.Series([0 for date in date_range], index=date_range)
    tweet_count.update(counts)

    # performing the grouping on the tweet data frame
    tweets_df["date"] = pd.to_datetime(tweets_df["date"])
    tweets_df.set_index("date", inplace=True,
                        verify_integrity=False, drop=False)
    tweets_df.sort_index(inplace=True, ascending=True)
    tweets_df = tweets_df.groupby(pd.Grouper(freq="D")).sum()
    tweets_df.index.names = ["index"]
    tweets_df["date"] = tweets_df.index

    # making sure that the tweet and stocks data frame end at the same day
    tweets_df = tweets_df.truncate(before=starting_date,
                                   after=ending_date, axis=0)
    stocks_df = stocks_df.truncate(before=starting_date,
                                   after=ending_date, axis=0)

    # adding count to the tweets_df
    tweet_count = tweet_count.truncate(before=starting_date,
                                       after=ending_date, axis=0)
    tweets_df["ntweets"] = tweet_count

    return stocks_df, tweets_df, date_range


    """ Creating the grouped DataFrames and ColumnDataSources """
def group_by_year(stocks_df, tweets_df):
    # group by years
    stocks_year = stocks_df.groupby(stocks_df["date"].dt.year).mean()
    counts_year = tweets_df.groupby(tweets_df["date"].dt.year).sum()
    combined_year = pd.concat([stocks_year, counts_year], axis=1)
    combined_year.index = [pd.Timestamp(f"{year}-01-01")
                           for year in combined_year.index]
    combined_year["date"] = combined_year.index
    combined_year = combined_year.assign(colour="black")
    return combined_year

def group_by_season(stocks_df, tweets_df):
    # group by months (not differentiating between years)
    stocks_season = stocks_df.groupby(stocks_df["date"].dt.month).mean()
    counts_season = tweets_df.groupby(tweets_df["date"].dt.month).sum()
    combined_season = pd.concat([stocks_season, counts_season], axis=1)
    combined_season.index = [pd.Timestamp(f"2022-{month}-01")
                             for month in combined_season.index]
    combined_season["date"] = combined_season.index 
    combined_season = combined_season.assign(colour="black")
    return combined_season

def group_by_month(stocks_df, tweets_df):
    # group by year and months
    stocks_month = stocks_df.groupby(pd.Grouper(freq="M")).mean()
    counts_month = tweets_df.groupby(pd.Grouper(freq="M")).sum()
    combined_month = pd.concat([stocks_month, counts_month], axis=1)
    combined_month["date"] = combined_month.index
    combined_month = combined_month.assign(colour="black")
    return combined_month

def group_by_week(stocks_df, tweets_df):
    # group by weeks
    stocks_week = stocks_df.groupby(pd.Grouper(freq='W-MON')).mean()
    counts_week = tweets_df.groupby(pd.Grouper(freq='W-MON')).sum()
    combined_week = pd.concat([stocks_week, counts_week], axis=1)
    combined_week["date"] = combined_week.index
    combined_week = combined_week.assign(colour="black")
    return combined_week

def group_by_day(stocks_df, tweets_df):
    # ungrouped/grouped by day
    stocks_df.drop(columns=["date"], inplace=True)
    combined_df = pd.concat([tweets_df, stocks_df], axis=1)
    combined_df = combined_df.assign(colour="black")
    return combined_df

def full_grouping(tweet_columns=None,
                  starting_date=None,
                  ending_date=None):

    stocks, tweets, date_range = make_DataFrames(tweet_columns,
                                                 starting_date,
                                                 ending_date)
    year = group_by_year(stocks, tweets)
    season = group_by_season(stocks, tweets)
    month = group_by_month(stocks, tweets)
    week = group_by_week(stocks, tweets)
    day = group_by_day(stocks, tweets)

    return day, week, month, year, season, date_range, stocks


if __name__ == "__main__":
    tweet_columns = ["date", "nlikes", "nreplies", "nretweets"]

    start = "2012-10-12"
    end = "2021-12-31"
    stocks, tweets, daterange = make_DataFrames(tweet_columns, None, None)


    week = group_by_day(stocks, tweets)
    print(week)