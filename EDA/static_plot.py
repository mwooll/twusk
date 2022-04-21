import numpy as np
import pandas as pd

import bokeh.palettes as bp
from bokeh.plotting import figure
from bokeh.io import output_file, show, save
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, RangeTool
from bokeh.transform import log_cmap
from bokeh.layouts import gridplot



""" Data Preprocessing """


"""Reading the wanted columns of the tweets data into a dataframe."""
tweets_file_name = "data/needed/cleaned/Tweets.csv"
needed_columns = ["date"] #, "hashtags", "cashtags", "nlikes", "nreplies", "nretweets"]
tweets_df = pd.read_csv(tweets_file_name, index_col=None,
                        usecols=needed_columns)

tweets_df["date"] = pd.to_datetime(tweets_df["date"]).dt.date


"""Creating a dataframe which contains the number of tweets per day"""
starting_date = min(tweets_df["date"])
ending_date = max(tweets_df["date"])

date_range = pd.date_range(start=starting_date, end=ending_date, freq="D")
counts = tweets_df["date"].value_counts()
counts.sort_index(inplace=True)

tweet_count = pd.Series([0 for date in date_range], index=date_range)
tweet_count.update(counts)
tweet_count_df = pd.DataFrame({"date": date_range, "count": tweet_count})
print(tweet_count_df)


"""Reading the stock data into a dataframe."""
stocks_file_name = "data/needed/cleaned/Stock.csv"
stocks_df = pd.read_csv(stocks_file_name, index_col=None)

stocks_df["date"] = pd.to_datetime(stocks_df["date"])
print(stocks_df)

"""Creating ColumnDataSources for plotting."""
stock_source = ColumnDataSource(stocks_df)
tweet_count_source = ColumnDataSource(tweet_count_df)
# tweet_source = ColumnDataSource()


"""Calculating the necessary maxima to set the plot heights."""
max_closing = np.max(stocks_df["close"])
max_tweet_count = np.max(tweet_count_df["count"])

"""Making a colour map."""
max_volume = np.max(stocks_df["volume"])
min_volume = np.min(stocks_df["volume"])
mapper = log_cmap(field_name="volume", palette=bp.Viridis256,
                  low=min_volume, high=max_volume)



""" Data Visualization """


"""Creating a scatter plot showing the closing prices."""
TOOLS = "box_select,lasso_select,wheel_zoom,pan,reset,help"
plot_stock = figure(plot_width=1200, plot_height=500,
                    tools=TOOLS, x_axis_type="datetime",
                    x_range=[date_range[0], date_range[365]],
                    y_range=[0, max_closing*1.1])
plot_stock.scatter(x="date", y="close", size=7,
                   marker="circle", source=stock_source , color=mapper)

plot_stock.title.text = "TSLA Closing Prices"
plot_stock.yaxis.axis_label = "Closing Price [$]"
plot_stock.xaxis.axis_label = "Date"
plot_stock.sizing_mode = "stretch_both"


"""Adding a HoverTool."""
hover_stock = HoverTool(tooltips = [
                ("date", "@date{%F}"),
                ("close", "@close$"),
                ("open", "@open$"),
                ("low", "@low$"),
                ("high", "@high$"),
                ("volume", "@volume{0,0}$")],
            formatters={'@date': 'datetime'})
plot_stock.add_tools(hover_stock)


"""Creating a ColorBar and adding it to the stock scatter plot."""
color_bar = ColorBar(color_mapper=mapper["transform"], width=10, location=(0,0))
plot_stock.add_layout(color_bar, "right")


"""Creating a line plot showing the number of tweets."""
plot_tweet = figure(plot_width=1200, plot_height=200,
                    x_axis_type="datetime",
                    y_range=[0, max_tweet_count*1.1], tools="")
plot_tweet.title.text = "Drag the middle and edges of the selection box to change the range above."
plot_tweet.yaxis.axis_label = "Number of Tweets"
plot_tweet.xaxis.axis_label = "Date"
plot_tweet.sizing_mode = "stretch_width"

"""Defining a RangeTool which is linked to the x_range in the scatter plot."""
range_tool = RangeTool(x_range=plot_stock.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2


"""Plotting number of tweets and adding the RangeTool to the plot."""
plot_tweet.line(x="date", y="count", source=tweet_count_source)
plot_tweet.add_tools(range_tool)


"""Adding a hovertool which displays date and the number of tweets."""
hover_tweets = HoverTool(tooltips = [
                 ("date", "@date{%F}"),
                 ("number of tweets", "@count")],
             formatters={'@date': 'datetime'})
plot_tweet.add_tools(hover_tweets)



"""Arranging, showing the layout."""
linked_p = gridplot(children=[[plot_stock], [plot_tweet]])
linked_p.sizing_mode = "stretch_both"
show(linked_p)
output_file("closing_price_per_day.html")
save(linked_p)