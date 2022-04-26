import numpy as np
import pandas as pd

import bokeh.palettes as bp
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, RangeTool
from bokeh.transform import log_cmap

from bokeh.models import Div, Select, Button, RadioButtonGroup, Slider
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row

from sklearn import cluster

""" Setting parameters """
# plot dimensions
left_width = 1200

large_height = 500
small_height = 200

right_width = 600
small_width = right_width//3 - 7 # should be at least 150
tiny_width = 100

# glyph attributes
dot_size = 5
# alpha_value = 0.5
line_width = 0.5


""" Global objects """
plot_names = ["upper", "lower", "scatter"]
plotted = {"upper": "close", "scatter-x": "open", "scatter-y": "close"}
cluster_colours = ["red", "blue", "green", "purple", "orange", "cyan", "lime"]

grouping = {"keys": ["day", "month", "year", "season"], "active": "day"}
dataframes = {key: None for key in grouping["keys"]}
dataframes["raw stocks"] = None
source_dict = {key: None for key in grouping["keys"]}
# ranges_dict = {key: None for key in grouping["keys"]}
cluster_dict = {key: {} for key in grouping["keys"]}

""" Callback functions """
def update_upper_time_plot():
    """update funtion for the upper time plot"""
    data, denom, form = get_select_values(stock_data,
                                          stock_ratio,
                                          stock_format)

    # updating the df and the source, and getting the column-name to plot
    name = update_dataframe_and_source(data, denom, form)

    # updating the plot
    plot_stock.renderers = []
    plot_stock.scatter(x="date", y=name, color="colour",
                       size=dot_size, marker="circle", 
                       source=source_dict[grouping["active"]])
    plotted["upper"] = name

    # updating the axes-range and the y-axis label
    set_axis_range("date", plot_stock.x_range)
    set_axis_range(name, plot_stock.y_range)
    set_axis_label(data, denom, form, plot_stock.yaxis)

def update_lower_time_plot( ): #attr, old, new):
    """update function for the lower time plot"""
    name = tweet_select.value

    # updating the line
    plot_tweet.renderers = []
    plot_tweet.line(x="date", y=name,
                    line_width=line_width,
                    source=source_dict[grouping["active"]])

    # updating the range and the label
    set_axis_range(name, plot_tweet.y_range)
    # set_axis_label(name, plot_tweet.yaxis)


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

    # updating the df and the source, and getting the column-names to plot
    x_name = update_dataframe_and_source(x_data, x_denom, x_form)
    y_name = update_dataframe_and_source(y_data, y_denom, y_form)

    # updating the plot
    scatter_plot.renderers = []
    scatter_plot.scatter(x=x_name, y=y_name, color="colour",
                         size=dot_size, marker="circle", 
                         source=source_dict[grouping["active"]])
    plotted["scatter-x"] = x_name
    plotted["scatter-y"] = y_name

    # updating the range and the label
    # of the x-axis
    set_axis_range(x_name, scatter_plot.x_range)
    set_axis_label(x_data, x_denom, x_form, scatter_plot.xaxis)
    # of the y-axis
    set_axis_range(y_name, scatter_plot.y_range)
    set_axis_label(y_data, y_denom, y_form, scatter_plot.yaxis)

def perform_grouping(new):
    """changing the source to be plotted and make the necessary changes"""
    # fetching the new class to group by
    for i, key in enumerate(grouping["keys"]):
        if i == new:
            grouping["active"] = key

    # updating the plots
    update_upper_time_plot()
    update_lower_time_plot()
    update_scatter_plot()

def perform_clustering():
    """performing clustering on the data"""
    # fetching the values from the select widgets
    cluster_k = int(clustering_k.value)
    method = cluster_method.value
    attributes = cluster_attributes.value
    random_state = cluster_random.value

    # gathering what attributes to consider when clustering
    if attributes == "visible on scatter plot":
        to_cluster = sorted([plotted["scatter-x"], plotted["scatter-y"]])

    # preparing the data on which we perform the clustering
    identifier = " ".join(to_cluster)
    if identifier not in cluster_dict[grouping["active"]].keys():
        cluster_data = dataframes[grouping["active"]].loc[:, to_cluster]
        cluster_data.dropna(axis=0, how="any", inplace=True)
        cluster_dict[grouping["active"]][identifier] = cluster_data

    # performing the clustering
    if method == "k-means":
        kmeans = cluster.KMeans(n_clusters=cluster_k, copy_x=False,
                                random_state=random_state, max_iter=500)
        kmeans.fit_predict(cluster_dict[grouping["active"]][identifier])
        labels = kmeans.labels_
    elif method != "k-means":
        print("The chosen method has not been implmented yet.")

    # updating the dataframe and source with the new colour column
    if grouping["active"] == "day":
        colour = [cluster_colours[k] for k in labels]
        try:
            short = pd.Series(colour, index=dataframes["raw stocks"].index)
            series = pd.Series([short[k] if k in short.index else "black"
                                for k in dataframes["day"].index],
                                index = dataframes["day"].index)
        except ValueError:
            print('"colour" was longer then the index of "raw stocks"')
            series = pd.Series(colour, index=dataframes["day"].index)
    else:
        series = pd.Series([cluster_colours[k] for k in labels],
                           index=dataframes[grouping["active"]].index)

    dataframes[grouping["active"]]["colour"] = series
    source_dict[grouping["active"]].data = dict(dataframes[grouping["active"]])


""" Helper functions """
def get_select_values(data_select, ratio_select, format_select):
    # fetching the values from the select widgets
    data = data_select.value
    ratio = ratio_select.value
    form = format_select.value

    # making sure that data is a valid entry in the dataframe/source
    if data == "number of tweets":
        # print('caught "number of tweets".')
        data = "count"

    # fetching the denominator, if any
    if ratio == "data":
        denom = None
    else:
        denom = ratio[5:]

    return data, denom, form

def update_dataframe_and_source(data, denom, form):
    # calculating the needed data if it isn't already in the dataframe
    if denom is not None:
        ratio = f"{data}_{denom}"
        if ratio not in dataframes[grouping["active"]].columns:
            column = (dataframes[grouping["active"]][data]
                      / dataframes[grouping["active"]][denom])
            dataframes[grouping["active"]][ratio] = column
    else:
        ratio = data

    name = ratio
    if form != "value":
        name = f"{ratio}_{form}"
        if name not in dataframes[grouping["active"]].columns:
            absolute = np.absolute(dataframes[grouping["active"]][ratio])
            dataframes[grouping["active"]][f"{ratio}_absolute"] = absolute
            logs = [np.log(k) if k > 0 else -10 for k in absolute]
            dataframes[grouping["active"]][f"{ratio}_log_10(absolute)"] = logs

    # updating the ColumnDataSource
    source_dict[grouping["active"]].data = dict(dataframes[grouping["active"]])

    # returning name so we know what to plot
    return name

def set_axis_range(name, axis):
    min_val = np.min(dataframes[grouping["active"]][name])
    max_val = np.max(dataframes[grouping["active"]][name])
    buffer = (max_val - min_val)*0.05
    axis.start = min_val - buffer
    axis.end = max_val + buffer

def set_axis_label(data, denom, form, axis):
    label = f"{data}"
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


""" Preparing the Data """
def make_DataFrames_and_ColumnDataSources():
    """ Reading in the data """
    # Reading the wanted columns of the tweets data into a dataframe.
    tweets_file_name = "../data/needed/cleaned/Tweets.csv"
    needed_columns = ["date"]#, "tweet", "hashtags", "cashtags",
                      # "nlikes", "nreplies", "nretweets"]
    tweets_df = pd.read_csv(tweets_file_name, index_col=None,
                            usecols=needed_columns)

    tweets_df["date"] = pd.to_datetime(tweets_df["date"]).dt.date

    # Reading the stock data into a dataframe.
    stocks_file_name = "../data/needed/cleaned/Stock.csv"
    stocks_df = pd.read_csv(stocks_file_name, index_col=None)
    stocks_df["date"] = pd.to_datetime(stocks_df["date"])
    stocks_df.set_index("date", inplace=True,
                        verify_integrity=False, drop=True)
    columns = list(stocks_df.columns)
    stocks_df["close - open"] = stocks_df["close"] - stocks_df["open"]
    col_with = list(stocks_df.columns) + ["number of tweets"]
    stocks_df.index.names = ["index"]
    stocks_df["date"] = stocks_df.index
    dataframes["raw stocks"] = stocks_df
    # print(stocks_df)

    # Counting how many tweets there are for each day
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
    tweet_count_df.set_index("date", inplace=True,
                             verify_integrity=False, drop=False)
    tweet_count_df.sort_index(inplace=True, ascending=False)
    # print(tweet_count_df)


    """ Creating the grouped DataFrames and ColumnDataSources """
    # group by years
    stocks_year = stocks_df.groupby(stocks_df["date"].dt.year).mean()
    counts_year = tweet_count_df.groupby(tweet_count_df["date"].dt.year).sum()
    combined_year = pd.concat([stocks_year, counts_year], axis=1)
    combined_year.index = [pd.Timestamp(f"{year}-01-01")
                           for year in combined_year.index]
    combined_year["date"] = combined_year.index
    combined_year = combined_year.assign(colour="black")
    source_year = ColumnDataSource(combined_year)

    dataframes["year"] = combined_year
    source_dict["year"] = source_year
    # print(combined_year)

    # group by months (not differentiating between years)
    stocks_season = stocks_df.groupby(stocks_df["date"].dt.month).mean()
    counts_season = tweet_count_df.groupby(tweet_count_df["date"].dt.month).sum()
    combined_season = pd.concat([stocks_season, counts_season], axis=1)
    combined_season.index = [pd.Timestamp(f"2022-{month}-01")
                             for month in combined_season.index]
    combined_season["date"] = combined_season.index 
    combined_season = combined_season.assign(colour="black")
    source_season = ColumnDataSource(combined_season)

    dataframes["season"] = combined_season
    source_dict["season"] = source_season
    # print(combined_season)

    # group by year and months
    stocks_month = stocks_df.groupby(pd.Grouper(freq="M")).mean()
    counts_month = tweet_count_df.groupby(pd.Grouper(freq="M")).sum()
    combined_month = pd.concat([stocks_month, counts_month], axis=1)
    combined_month.index.names = ["index"]
    combined_month["date"] = combined_month.index
    combined_month = combined_month.assign(colour="black")
    source_month = ColumnDataSource(combined_month)

    dataframes["month"] = combined_month
    source_dict["month"] = source_month
    # print(combined_month)

    # group by weeks 

    # ungrouped/grouped by day
    stocks_df.drop(columns=["date"], inplace=True)
    combined_df = pd.concat([tweet_count_df, stocks_df], axis=1)
    combined_df.index.names = ["index"]
    combined_df = combined_df.assign(colour="black")
    source_day = ColumnDataSource(combined_df)

    dataframes["day"] = combined_df
    source_dict["day"] = source_day
    # print(combined_df)

    return date_range, columns, col_with

date_range, columns, col_with = make_DataFrames_and_ColumnDataSources()

""" Getting the initial ranges and the colour mapper """
# Calculating the necessary maxima to set the plot heights.
max_closing = np.max(dataframes[grouping["active"]]["close"])
max_opening = np.max(dataframes[grouping["active"]]["open"])
max_tweet_count = np.max(dataframes[grouping["active"]]["count"])

# Making a colour mapper.
max_volume = np.max(dataframes[grouping["active"]]["volume"])
min_volume = np.min(dataframes[grouping["active"]]["volume"])
mapper = log_cmap(field_name="volume", palette=bp.Viridis256,
                  low=min_volume, high=max_volume)



""" Making the left column """

""" Creating the upper time plot. """
TOOLS = "box_select, lasso_select, box_zoom, wheel_zoom, pan, reset"
plot_stock = figure(plot_width=left_width, plot_height=large_height,
                    tools=TOOLS, toolbar_location="above",
                    # x_axis_type="datetime",
                    x_range=[date_range[0], date_range[-1]],
                    y_range=[0, max_closing*1.1])
plot_stock.scatter(x="date", y="close", color="colour",
                   size=dot_size, marker="circle",
                   source=source_dict[grouping["active"]])

# plot_stock.title.text = "TSLA Closing Prices"
plot_stock.yaxis.axis_label = "close"
plot_stock.sizing_mode = "stretch_both"

# Adding a HoverTool to the upper time plot.
stocks_hover = HoverTool(tooltips = [
                    ("date", "@date{%F}"),
                    ("close", "@close$"),
                    ("open", "@open$"),
                    ("low", "@low$"),
                    ("high", "@high$"),
                    ("volume", "@volume{0,0}$"),
                    ("close - open", "@{close - open}$")],
                formatters={'@date': 'datetime'})
plot_stock.add_tools(stocks_hover)

# =============================================================================
# # Creating a ColorBar and adding it to the upper time plot.
# color_bar = ColorBar(color_mapper=mapper["transform"],
#                      width=10, location=(0,0))
# plot_stock.add_layout(color_bar, "right")
# =============================================================================


""" Creating the lower time plot. """
plot_tweet = figure(plot_width=left_width, plot_height=small_height,
                    tools="", toolbar_location=None,
                    x_axis_type="datetime",
                    y_range=[0, max_tweet_count*1.1])
explanation = "drag the middle and edges of the box to change the range above."
plot_tweet.title.text = explanation
plot_tweet.yaxis.axis_label = "number of tweets"
plot_tweet.xaxis.axis_label = "date"
plot_tweet.sizing_mode = "stretch_width"

plot_tweet.line(x="date", y="count", source=source_dict[grouping["active"]],
                line_width=line_width)

# Defining a RangeTool which is linked to the x_range of the upper time plot.
range_tool = RangeTool(x_range=plot_stock.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2
plot_tweet.add_tools(range_tool)


# Adding a hovertool to the lower time plot.
tweets_hover = HoverTool(tooltips = [
                    ("date", "@date{%F}"),
                    ("number of tweets", "@count")],
                formatters={'@date': 'datetime'})
plot_tweet.add_tools(tweets_hover)


""" Making the change row for the time plots """
stock_data = Select(title="data", value="close",
                          options=col_with, width=small_width)

stock_ratio = Select(title="ratio", value="data",
                      options=["data"] + [f"data/{col}" for col in columns],
                      width=small_width)

stock_format = Select(title="format", value="value",
                       options=["value", "absolute", "log_10(absolute)"],
                       width=small_width)

update_upper_button = Button(label="apply changes to the upper plot")
update_upper_button.on_click(update_upper_time_plot)

tweet_select = Select(title="tweet data", value="count",
                       options=["count", "nlikes", "nretweets", "nreplies"])
# tweet_select.on_change("value", update_lower_time_plot)
tweet_select.disabled = True

""" Making the right column """
scatter_plot = figure(plot_width=right_width, plot_height=right_width,
                      tools=TOOLS,  toolbar_location="above",
                      x_range=[-100, max_closing*1.1],
                      y_range=[-100, max_opening*1.1])
# scatter_plot.sizing_mode = "stretch_both"
scatter_plot.xaxis.axis_label = "open"
scatter_plot.yaxis.axis_label = "close"
scatter_plot.scatter(x="open", y="close", color="colour",
                     size=dot_size, marker="circle",
                     source=source_dict[grouping["active"]])

# Adding a HoverTool to the scatter plot.
scatter_hover = HoverTool(tooltips = [
                        ("date", "@date{%F}"),
                        ("open", "@open$"),
                        ("close", "@close$")],
                    formatters={'@date': 'datetime'})
scatter_plot.add_tools(scatter_hover)


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

update_scatter_button = Button(label="apply changes to the scatter plot",
                               width=right_width)
update_scatter_button.on_click(update_scatter_plot)


""" Making the RadioButtonGroup for the grouping """
grouping_div = Div(text="choose how the data entries should be grouped",
                   width=right_width)
button_group = RadioButtonGroup(labels=grouping["keys"])
button_group.on_click(perform_grouping)


""" Making the widgets for the clustering """
clustering_k = Slider(title="choose the number of clusters",
                        start=1, end=7, value=3, step=1,
                        width=small_width)
cluster_method = Select(title="choose the clustering method", value="k-means",
                        options=["k-means", "k-medoids",
                                 "bottom-up", "top-down"],
                        width=small_width)
cluster_method.disabled = True
cluster_attributes = Select(title="choose the attributes",
                            value="visible on scatter plot",
                            options=["visible on scatter plot"],
                            width=small_width)
cluster_attributes.disabled = True
cluster_random = Slider(title="choose the random state",
                        start=0, end=10, value=0, step=1,
                        width=small_width)
cluster_norm = Select(title="normalize data", value="false",
                      options=["false", "true"], width=tiny_width)
cluster_button = Button(label="perform clustering", width=small_width)
cluster_button.on_click(perform_clustering)
cluster_div = Div(text="date won't be considered when performing clustering.",
                  width=right_width)


""" Adding a Select for data imputation """
impatation_select = Select




""" Arranging the layout and showing it """
left_column = column(plot_stock,
                     plot_tweet,
                     row(stock_data,
                         stock_ratio,
                         stock_format,
                         update_upper_button,
                         tweet_select))
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

plot_part = row(left_column, right_column)
cluster_part = row(clustering_k,
                   cluster_method,
                   cluster_attributes,
                   cluster_random,
                   cluster_norm,
                   cluster_button,
                   # cluster_div
                   )

layout = column(plot_part, cluster_part)
layout.sizing_mode = "stretch_both"
curdoc().add_root(layout)
curdoc().title = "Interactive explorer"