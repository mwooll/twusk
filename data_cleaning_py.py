# load necessary libraries
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


"""Processing Tweets"""

# load data sets
tweets_2020 = pd.read_csv('data/needed/raw/2020.csv', index_col=0)
tweets_2021 = pd.read_csv('data/needed/raw/2021.csv')
tweets_2022 = pd.read_csv('data/needed/raw/2022.csv')

# tweets_2022 contains duplicates with very slightly different likes,
# replies and retweet counts; we remove them
tweets_2022 = tweets_2022.drop_duplicates(subset=["id"], keep="last")

# find differences in columns
print(set(tweets_2020)-(set(tweets_2021)))
print(set(tweets_2021)-(set(tweets_2020)))
print(set(tweets_2021).symmetric_difference(set(tweets_2022)))

# make column names match between dataframes
tweets_2020 = tweets_2020.rename(columns = {"day":"day_of_week", "hour":"hour_of_day"})
tweets_2021 = tweets_2021.rename(columns = {"retweets_count":"nretweets", "replies_count":"nreplies", "likes_count":"nlikes"})
tweets_2022 = tweets_2022.rename(columns = {"retweets_count":"nretweets", "replies_count":"nreplies", "likes_count":"nlikes"})

# make one column with date and time
tweets_2021["date"] = tweets_2021["date"] + " " + tweets_2021["time"]
tweets_2021 = tweets_2021.drop(["time"], 1)

tweets_2022["date"] = tweets_2022["date"] + " " + tweets_2022["time"]
tweets_2022 = tweets_2022.drop(["time"], 1)

# convert the date columns to datetime
for i in [tweets_2020, tweets_2021, tweets_2022]:
    i["date"] = pd.to_datetime(i["date"])

# add columns that were in one data frame but not the others
tweets_2021["day_of_week"] = tweets_2021["date"].dt.dayofweek + 1
tweets_2022["day_of_week"] = tweets_2022["date"].dt.dayofweek + 1

tweets_2021["hour_of_day"] = tweets_2021["date"].dt.hour
tweets_2022["hour_of_day"] = tweets_2022["date"].dt.hour


# check that

# make one df out of all 3 datasets
tweets = tweets_2020.append(tweets_2021, ignore_index= True)
tweets = tweets.append(tweets_2022, ignore_index= True)
tweets = tweets.sort_values(by=['date'], ascending=False)
print(tweets.shape)


# remove all empty columns

#tweets.info()
tweets = tweets.dropna(1, how="all")
#tweets.info()

# save tweets df to csv
tweets.to_csv('data/needed/cleaned/Tweets.csv', index=False)



"""Stock Price"""

# load data sets
stock_2018 = pd.read_csv('data/needed/raw/TSLA_2010_2018.csv')
stock_2022 = pd.read_csv('data/needed/raw/TSLA_2017_2022.csv')

# we drop "Adj Close" since it is a duplicate of "Close"
stock_2018 = stock_2018.drop(["Adj Close"], 1)

# rename "Close/Last" to "Close" to be able to merge the data frames
stock_2022 = stock_2022.rename(columns = {"Close/Last":"Close"})

# convert the date columns to datetime
stock_2022["Date"] = pd.to_datetime(stock_2022["Date"], format="%m/%d/%Y")
stock_2018["Date"] = pd.to_datetime(stock_2018["Date"], format="%Y-%m-%d")

# remove all dollar signs before values
for i in ["Close", "Open", "High", "Low"]:
    stock_2022[i] = stock_2022[i].str.replace("$","")

# check that data frames have compatible columns
columns_2018 = stock_2018.columns
columns_2022 = stock_2022.columns
try:
    assert columns_2018 == columns_2022
except AssertionError:
    raise AssertionError("The data seems to be corrupted." +
                         "Column names do not align, "+
                         "cannot merge data frames containing stock prices.")

# put the 2 df's together and remove all duplicates
stock = stock_2018.append(stock_2022, ignore_index=True)
stock = stock.drop_duplicates(subset=["Date"])

# make all column names lowercase
stock.columns = stock.columns.str.lower()

# save stock df to csv
stock = stock.sort_values(by=['Date'], ascending=False)
stock.to_csv('data/needed/cleaned/Stock.csv', index=False)

#df with all the data
tweets["date"] = tweets["date"].dt.date
stock["date"] = stock["date"].dt.date
together = pd.merge(tweets, stock, on = ["date"], how="outer")
together = together.sort_values(by=['date'], ascending=False)
together.to_csv('data/needed/cleaned/Tweets_and_Stock.csv', index=False)