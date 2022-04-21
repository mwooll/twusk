import numpy as np
import pandas as pd

import bokeh.palettes as bp
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, RangeTool
from bokeh.transform import log_cmap

from bokeh.models import Div, Select, Button, RadioButtonGroup #Slope 
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row


""" Setting parameters """
# plot dimensions
left_width = 1200

large_height = 500
small_height = 200

right_width = 600
small_width = right_width//3 - 7 # should be at least 150

# glyph attributes
dot_size = 5
alpha_value = 0.5
line_width = 0.5


""" Callback functions """
def update_upper_time_plot():
    """update funtion for the upper time plot"""
    data, denom, form = get_select_values(stock_data_select,
                                          ratio_select,
                                          format_select)

    # updating the df and the source, and getting the column-name to plot
    name = update_combined_df_and_source(data, denom, form)


    # updating the plot
    plot_stock.renderers = []
    plot_stock.scatter(x="date", y=name,
                       size=7, marker="circle",
                       source=source , color=mapper)

    # updating the range and the label of the y-axis
    set_axis_range(name, plot_stock.y_range)
    set_axis_label(data, denom, form, plot_stock.yaxis)

def update_scatter_plot():
    """update funtion for the scatter plot"""
    # fetching the values from the select widgets
    # for the x-axis
    x_data, x_denom, x_form = get_select_values(scatter_x_data,
                                                scatter_x_ratio,
                                                scatter_x_format)
    # for the y-axis
    y_data, y_denom, y_form = get_select_values(scatter_y_data,
                                                scatter_y_ratio,
                                                scatter_y_format)
    print(x_data, x_denom, x_form)
    print(y_data, y_denom, y_form)

    # updating the df and the source, and getting the column-names to plot
    x_name = update_combined_df_and_source(x_data, x_denom, x_form)
    y_name = update_combined_df_and_source(y_data, y_denom, y_form)

    # updating the plot
    scatter_plot.renderers = []
    scatter_plot.scatter(x=x_name, y=y_name,
                         size=dot_size, marker="circle",
                         source=source, color=mapper)

    # updating the range and the label
    # of the x-axis
    set_axis_range(x_name, scatter_plot.x_range)
    set_axis_label(x_data, x_denom, x_form, scatter_plot.xaxis)
    # of the y-axis
    set_axis_range(y_name, scatter_plot.y_range)
    set_axis_label(y_data, y_denom, y_form, scatter_plot.yaxis)

def perform_grouping():
    pass

""" Helper functions """
def get_select_values(data_select, ratio_select, format_select):
    # fetching the values from the select widgets
    data = data_select.value
    ratio = ratio_select.value
    form = format_select.value

    # fetching the denominator, if any
    if ratio == "data":
        denom = None
    else:
        denom = ratio[5:]

    return data, denom, form

def update_combined_df_and_source(data, denom, form):
    # calculating the needed data if it isn't already in the dataframe
    if denom is not None:
        ratio = f"{data}_{denom}"
        if ratio not in combined_df.columns:
            combined_df[ratio] = combined_df[data]/combined_df[denom]
    else:
        ratio = data

    name = ratio
    print(f"name = '{name}'.")
    if form != "value":
        name = f"{ratio}_{form}"
        if name not in combined_df.columns:
            absolute = np.absolute(combined_df[ratio])
            combined_df[f"{ratio}_absolute"] = absolute
            logs = [np.log(k) if k > 0 else -10 for k in absolute]
            combined_df[f"{ratio}_log_10(absolute)"] = logs
    print(f"name = '{name}'.")

    # updating the ColumnDataSource
    source.data = dict(combined_df)
    print(combined_df.columns)
    print(source.data[name])

    # returning name so we know what to plot
    return name

def set_axis_range(name, axis):
    min_val, max_val = np.min(combined_df[name]), np.max(combined_df[name])
    buffer = (max_val - min_val)*0.05
    axis.start = min_val - buffer
    axis.end = max_val + buffer

def set_axis_label(data, denom, form, axis):
    label = f"{data} [$]"
    if denom is not None:
        label = f"{data}/{denom}"
    if form != "value":
        label.replace(" [$]", "")
        if form == "absolute":
            label = f"abs({label})"
        if form == "log_10(absolute)":
            label = f"log_10(abs({label}))"
    axis.axis_label = label
    return label


""" Reading in the data """

# Reading the wanted columns of the tweets data into a dataframe.
tweets_file_name = "../data/needed/cleaned/Tweets.csv"
needed_columns = ["date", "tweet", "hashtags", "cashtags",
                  "nlikes", "nreplies", "nretweets"]
tweets_df = pd.read_csv(tweets_file_name, index_col=None,
                        usecols=needed_columns)

tweets_df["date"] = pd.to_datetime(tweets_df["date"]).dt.date

# Reading the stock data into a dataframe.
stocks_file_name = "../data/needed/cleaned/Stock.csv"
stocks_df = pd.read_csv(stocks_file_name, index_col=None)

stocks_df["date"] = pd.to_datetime(stocks_df["date"])
stocks_df.set_index("date", inplace=True, verify_integrity=False)
columns = list(stocks_df.columns)
stocks_df["close - open"] = stocks_df["close"] - stocks_df["open"]
col_with = list(stocks_df.columns)
# print(stocks_df)

# Creating a combined dataframe
tweet_min, tweet_max = min(tweets_df["date"]), max(tweets_df["date"])
stock_min, stock_max = max(stocks_df.index), max(stocks_df.index)
starting_date = min(pd.Timestamp(tweet_min), pd.Timestamp(stock_min))
ending_date = max(pd.Timestamp(tweet_max), pd.Timestamp(stock_max))

date_range = pd.date_range(start=starting_date, end=ending_date, freq="D")
counts = tweets_df["date"].value_counts()
counts.sort_index(inplace=True)

tweet_count = pd.Series([0 for date in date_range], index=date_range)
tweet_count.update(counts)
tweet_count_df = pd.DataFrame({"date": date_range, "count": tweet_count})
tweet_count_df.set_index("date", inplace=True, verify_integrity=False)
tweet_count_df.sort_index(inplace=True, ascending=False)
# print(tweet_count_df)

combined_df = pd.concat([tweet_count_df, stocks_df], axis=1)
combined_df.index.names = ["index"]
combined_df["date"] = combined_df.index
source = ColumnDataSource(combined_df)
# print(combined_df)



# Calculating the necessary maxima to set the plot heights.
max_closing = np.max(combined_df["close"])
max_opening = np.max(combined_df["open"])
max_tweet_count = np.max(tweet_count_df["count"])

# Making a colour mapper.
max_volume = np.max(combined_df["volume"])
min_volume = np.min(combined_df["volume"])
mapper = log_cmap(field_name="volume", palette=bp.Viridis256,
                  low=min_volume, high=max_volume)


""" Making the left column """

""" Creating the upper time plot. """
TOOLS = "box_select, lasso_select, box_zoom, wheel_zoom, pan, reset, help"
plot_stock = figure(plot_width=left_width, plot_height=large_height,
                    tools=TOOLS, toolbar_location="above",
                    x_axis_type="datetime",
                    x_range=[date_range[0], date_range[365]],
                    y_range=[0, max_closing*1.1])
plot_stock.scatter(x="date", y="close",
                   size=dot_size, marker="circle",
                   source=source , color=mapper)

# plot_stock.title.text = "TSLA Closing Prices"
plot_stock.yaxis.axis_label = "close [$]"
plot_stock.sizing_mode = "stretch_both"

# Adding a HoverTool to the upper time plot.
hover_stock = HoverTool(tooltips = [
                    ("date", "@date{%F}"),
                    ("close", "@close$"),
                    ("open", "@open$"),
                    ("low", "@low$"),
                    ("high", "@high$"),
                    ("volume", "@volume{0,0}$"),
                    ("close - open", "@{close - open}$")],
                    formatters={'@date': 'datetime'})
plot_stock.add_tools(hover_stock)

# Creating a ColorBar and adding it to the upper time plot.
color_bar = ColorBar(color_mapper=mapper["transform"], width=10, location=(0,0))
plot_stock.add_layout(color_bar, "right")


""" Creating the lower time plot. """
plot_tweet = figure(plot_width=left_width, plot_height=small_height,
                    tools="", toolbar_location=None,
                    x_axis_type="datetime",
                    y_range=[0, max_tweet_count*1.1])
plot_tweet.title.text = "Drag the middle and edges of the box to change the range above."
plot_tweet.yaxis.axis_label = "Number of Tweets"
plot_tweet.xaxis.axis_label = "Date"
plot_tweet.sizing_mode = "stretch_width"

plot_tweet.line(x="date", y="count", source=source, line_width=line_width)

# Defining a RangeTool which is linked to the x_range of the upper time plot.
range_tool = RangeTool(x_range=plot_stock.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2
plot_tweet.add_tools(range_tool)


# Adding a hovertool to the lower time plot.
hover_tweets = HoverTool(tooltips = [
                     ("date", "@date{%F}"),
                     ("number of tweets", "@count")],
                     formatters={'@date': 'datetime'})
plot_tweet.add_tools(hover_tweets)


""" Making the change row for the upper time plot. """
stock_data_select = Select(title="data", value="close",
                          options=col_with, width=small_width)

ratio_select = Select(title="ratio", value="data",
                      options=["data"] + [f"data/{col}" for col in columns],
                      width=small_width)

format_select = Select(title="format", value="value",
                       options=["value", "absolute", "log_10(absolute)"],
                       width=small_width)

update_upper_button = Button(label="Apply changes to the upper plot")
update_upper_button.on_click(update_upper_time_plot)



""" Making the right column """
scatter_plot = figure(plot_width=right_width, plot_height=right_width,
                      tools=TOOLS,  toolbar_location="above",
                      x_range=[-100, max_closing*1.1],
                      y_range=[-100, max_opening*1.1])
# scatter_plot.sizing_mode = "stretch_both"
scatter_plot.xaxis.axis_label = "open [$]"
scatter_plot.yaxis.axis_label = "close [$]"
scatter_plot.scatter(x="open", y="close",
                     size=dot_size, marker="circle",
                     source=source, color=mapper)

# Adding a HoverTool to the scatter plot.
hover_scatter = HoverTool(tooltips = [
                        ("date", "@date{%F}"),
                        ("open", "@open$"),
                        ("close", "@close$")],
                        formatters={'@date': 'datetime'})
scatter_plot.add_tools(hover_scatter)


""" Making the change rows for the scatter plot. """
# selects for the x-axis
scatter_x_data = Select(title="x-data", value="open",
                        options=col_with, width=small_width)

scatter_x_ratio = Select(title="x-ratio", value="data",
                         options=["data"] + [f"data/{col}" for col in columns],
                         width=small_width)

scatter_x_format = Select(title="x-format", value="value",
                          options=["value", "absolute", "log_10(absolute)"],
                          width=small_width)

# selects for the y-axis
scatter_y_data = Select(title="y-data", value="close",
                        options=col_with, width=small_width)

scatter_y_ratio = Select(title="y-ratio", value="data",
                         options=["data"] + [f"data/{col}" for col in columns],
                         width=small_width)

scatter_y_format = Select(title="y-format", value="value",
                          options=["value", "absolute", "log_10(absolute)"],
                          width=small_width)

update_scatter_button = Button(label="Apply changes to the scatter plot",
                               width=right_width)
update_scatter_button.on_click(update_scatter_plot)


""" Making the RadioButtonGroup for the grouping """
grouping_div = Div(text="Choose how the data entries should be grouped.",
                   width=right_width)
button_group = RadioButtonGroup(labels=["Day", "Month", "Year"])
button_group.on_click(perform_grouping)

""" Arranging the layout and showing it """
left_column = column(plot_stock,
                     plot_tweet,
                     row(stock_data_select,
                         ratio_select,
                         format_select,
                         update_upper_button))
right_column = column(scatter_plot,
                      row(scatter_x_data,
                          scatter_x_ratio,
                          scatter_x_format),
                      row(scatter_y_data,
                          scatter_y_ratio,
                          scatter_y_format),
                      update_scatter_button,
                      grouping_div,
                      button_group)

layout = row(left_column, right_column)
layout.sizing_mode = "stretch_both"
curdoc().add_root(layout)
curdoc().title = "Interactive explorer"