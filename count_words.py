# counts total word occurences in all tweets
# does *not* count number of tweets where the words is included

import pandas as pd
import os

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
os.chdir(dir_path)

tweets = pd.read_csv('data/needed/cleaned/Tweets.csv')

tweets_text = tweets["tweet"]
tweets_text = tweets_text.to_list()
tweets_text = " ".join(tweets_text)
tweets_text = tweets_text.lower()

#symbols to ignore
for i in [",", ".", '”', "“", "(", ")", "!", "?", ":", ";", "[", "]", "*", "#", "@"]:
    tweets_text = tweets_text.replace(i, "")
tweets_text = tweets_text.replace("/", " ")

tweets_text = tweets_text.split(" ")
x = []
y = []

for i in tweets_text:
    if i in x:
        y[x.index(i)] += 1
    else:
        x.append(i)
        y.append(1)

word_count = pd.DataFrame(data={"word":x, "count":y})
word_count = word_count.sort_values(by=["count"], ascending=False)
word_count.to_csv('data/needed/cleaned/word_count.csv', index=False)