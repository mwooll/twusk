# interactive_explorer.py

import numpy as np
import pandas as pd
import datetime

from bokeh.models import (ColumnDataSource, HoverTool, RangeTool, ColorBar,
                          Div, Select, Button, RadioButtonGroup, Slider)
from bokeh.models.widgets import DateRangeSlider
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row

from sklearn import cluster
from sklearn_extra.cluster import KMedoids


from prepare_data import full_grouping


""" Setting parameters """
# plot dimensions
left_width = 1125

large_height = 550
small_height = 165

right_width = 550
small_width = right_width//3 - 7 # should be at least 150

tiny_width = 100
fill_width = left_width - right_width - 3*tiny_width - 36
button_margin = [23, 5, 0, 5]
grouping_margin = [10, 5, 0, 5]

# glyph attributes
dot_size = 5
line_width = 0.5

 # milliseconds in a day
length_of_day = 86400000

""" Global objects """
tweet_columns = ["date", "nlikes", "nreplies", "nretweets"]
plot_names = ["upper", "lower", "scatter"]
plotted = {"upper": "close", "scatter-x": "open", "scatter-y": "close"}
cluster_colours = ["red", "blue", "green", "purple", "orange", "cyan", "lime"]

grouping = {"keys": ["day", "week", "month", "year"], #, "season"],
            "active": "day"}
cluster_dict = {key: {} for key in grouping["keys"]}

""" update functions for the plots """
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

def update_lower_time_plot(attr, old, new):
    """update function for the lower time plot"""
    name = tweet_select.value

    # updating the line
    plot_tweet.renderers = []
    plot_tweet.line(x="date", y=name,
                    line_width=line_width,
                    source=source_dict[grouping["active"]])

    # updating the range and the label
    set_axis_range(name, plot_tweet.y_range)
    plot_tweet.yaxis.axis_label = name


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


""" Callback functions """
def perform_grouping(new):
    """changing the source to be plotted and make the necessary changes"""
    # fetching the new class to group by
    for i, key in enumerate(grouping["keys"]):
        if i == new:
            grouping["active"] = key

    # updating the plots
    update_upper_time_plot()
    update_lower_time_plot(None, None, None)
    update_scatter_plot()

def modify_clustering_k(attr, new, old):
    if new in ["affinity propagation"]:
        clustering_k.title = "preference"
    else:
        clustering_k.title = "number of clusters"

def perform_clustering():
    """performing clustering on the data"""
    # fetching the values from the select widgets
    cluster_k = int(clustering_k.value)
    method = cluster_method.value
    attributes = cluster_attributes.value
    normalize = cluster_norm.value

    # gathering what attributes to consider when clustering
    if attributes == "visible on scatter plot":
        to_cluster = sorted([plotted["scatter-x"], plotted["scatter-y"]])

    elif attributes == "open, close, volume, ntweets":
        to_cluster = ["close", "ntweets", "open", "volume"]

    # preparing the data on which we perform the clustering
    identifier = " ".join(to_cluster)
    if identifier not in cluster_dict[grouping["active"]].keys():
        cluster_data = dataframes[grouping["active"]].loc[:, to_cluster]
        cluster_data.dropna(axis=0, how="any", inplace=True)
        normalized = cluster_data.apply(lambda x: (x-x.mean())/x.std(), axis=0)
        cluster_dict[grouping["active"]][identifier] = {"false": cluster_data,
                                                        "true":  normalized}

    # performing the clustering
    data_to_cluster = cluster_dict[grouping["active"]][identifier][normalize]
    if method == "k-means":
        kmeans = cluster.KMeans(n_clusters=cluster_k,
                                copy_x=False, max_iter=500+100*cluster_k)
        kmeans.fit_predict(data_to_cluster)
        labels = kmeans.labels_

    elif method == "k-medoids":
        kmedoids = KMedoids(n_clusters=cluster_k, method="pam",
                            init="random", max_iter=500+100*cluster_k)
        kmedoids.fit(data_to_cluster)
        labels = kmedoids.labels_

    elif method in ["ward's method", "complete linkage", "single linkage"]:
        param = {"ward's method": "ward",
                 "complete linkage": "complete",
                 "single linkage": "single"}[method]
        agglom = cluster.AgglomerativeClustering(n_clusters=cluster_k,
                                                 linkage=param)
        labels = agglom.fit_predict(data_to_cluster)

    else:
        print("The chosen method has not been implemented yet.")
        return

    # updating the dataframe and source with the new colour column
    colour = [cluster_colours[k] for k in labels]
    data = dataframes[grouping["active"]]
    try:
        series = pd.Series(colour, index=data.index)
    except ValueError:
        custom_index = [data.loc[i]["date"] for i in data.index
                        if not np.isnan(data.loc[i]["close"])]
        short = pd.Series(colour, index=custom_index)
        series = pd.Series([short[k] if k in custom_index else k
                            for k in data.index],
                           index=data.index)


    dataframes[grouping["active"]]["colour"] = series
    source_dict[grouping["active"]].data["colour"] = series

def set_time_range():
    """ callback to alter the time range of the data used """
    start, end = date_slider.value
    start = start - start%length_of_day # to make sure we have full days
    end = end - end%length_of_day

    # getting timestamps from the slider values
    start_date = datetime.datetime.utcfromtimestamp(start/1000.0)
    end_date = datetime.datetime.utcfromtimestamp(end/1000.0)

    # truncating the dataframes to the chosen time range
    for grouping, data in full_data.items():
        truncated = data.truncate(before=start_date, after=end_date, axis=0)
        dataframes[grouping] = truncated
        cluster_dict[grouping] = {}

        if grouping != "raw stocks":
            source_dict[grouping] = ColumnDataSource(truncated)

    # updating the plots
    update_upper_time_plot()
    update_lower_time_plot(None, None, None)
    update_scatter_plot()    


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
            logs = [np.log(k) if k > 0 else -2
                    for k in absolute if not np.isnan(k)]

            log_key = f"{ratio}_log_10(absolute)"
            try:
                dataframes[grouping["active"]][log_key] = logs
            except ValueError:
                data = dataframes[grouping["active"]]
                custom_index = [data.loc[i]["date"] for i in data.index
                                    if not np.isnan(data.loc[i]["close"])]
                series = pd.Series(logs, index=custom_index)
                dataframes[grouping["active"]][log_key] = series

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
        if data == "close - open":
            data = "(close - open)"
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
data = full_grouping(tweet_columns)
day, week, month, year, season, date_range, raw = data

full_data = {"day": day, "week": week, "month": month,
              "year": year, "season": season}
source_dict = {key: ColumnDataSource(val) for key, val in full_data.items()}
full_data["raw stocks"] = raw
dataframes = {key: data for key, data in full_data.items()}
# we need a copy to be able to adjust the time range

# specifying the possible choices fo the data selects
stock_columns = list(raw.columns)
full_columns = tweet_columns + ["ntweets"] + stock_columns

""" Getting the initial ranges and the colour mapper """
# Calculating the necessary maxima to set the plot heights.
max_closing = np.max(dataframes[grouping["active"]]["close"])
max_opening = np.max(dataframes[grouping["active"]]["open"])
max_tweet_count = np.max(dataframes[grouping["active"]]["ntweets"])


""" Making the left column """

""" Creating the upper time plot """
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
plot_stock.xaxis.axis_label = "date"

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


""" Creating the lower time plot """
plot_tweet = figure(plot_width=left_width, plot_height=small_height,
                    tools="", toolbar_location=None,
                    x_axis_type="datetime",
                    y_range=[0, max_tweet_count*1.1],
                    margin=[-6, 0, 0, 0])
explanation = "drag the middle and edges of the box to change the range above."
plot_tweet.title.text = explanation
plot_tweet.yaxis.axis_label = "ntweets"
# plot_tweet.xaxis.axis_label = "date"

plot_tweet.line(x="date", y="ntweets", source=source_dict[grouping["active"]],
                line_width=line_width)

# Defining a RangeTool which is linked to the x_range of the upper time plot.
range_tool = RangeTool(x_range=plot_stock.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2
plot_tweet.add_tools(range_tool)

# Adding a hovertool to the lower time plot.
tweets_hover = HoverTool(tooltips = [
                    ("date", "@date{%F}"),
                    ("ntweets", "@ntweets"),
                    ("nlikes", "@nlikes"),
                    ("nreplies", "@nreplies"),
                    ("nretweet", "@nretweets")],
                formatters={'@date': 'datetime'})
plot_tweet.add_tools(tweets_hover)


""" Making the change row for the time plots """
stock_data = Select(title="data", value="close",
                          options=stock_columns, width=small_width)

stock_ratio = Select(title="ratio", value="data",
                      options=["data"] + [f"data/{col}"
                                          for col in stock_columns],
                      width=small_width)

stock_format = Select(title="format", value="value",
                       options=["value", "absolute", "log_10(absolute)"],
                       width=small_width)

update_upper_button = Button(label="apply changes to the upper plot",
                             margin=button_margin, width=3*tiny_width+10)
update_upper_button.on_click(update_upper_time_plot)

tweet_select = Select(title="tweet data", value="ntweets",
                       options=tweet_columns[1:]+["ntweets"],
                       width=fill_width)
tweet_select.on_change("value", update_lower_time_plot)


""" Making the right column """
scatter_plot = figure(plot_width=right_width, plot_height=right_width-16,
                      tools=TOOLS,  toolbar_location="above",
                      x_range=[-100, max_closing*1.1],
                      y_range=[-100, max_opening*1.1])
scatter_plot.xaxis.axis_label = "open"
scatter_plot.yaxis.axis_label = "close"
scatter_plot.scatter(x="open", y="close", color="colour",
                     size=dot_size, marker="circle",
                     source=source_dict[grouping["active"]])
scatter_plot.add_tools(stocks_hover)


""" Making the change rows for the scatter plot """
# selects for the x-axis
scatter_x_data = Select(title="x-data", value="open",
                        options=full_columns, width=small_width)

scatter_x_ratio = Select(title="x-ratio", value="data",
                         options=["data"] + [f"data/{col}"
                                             for col in stock_columns],
                         width=small_width)

scatter_x_format = Select(title="x-format", value="value",
                          options=["value", "absolute", "log_10(absolute)"],
                          width=small_width)

# selects for the y-axis
scatter_y_data = Select(title="y-data", value="close",
                        options=full_columns, width=small_width)

scatter_y_ratio = Select(title="y-ratio", value="data",
                         options=["data"] + [f"data/{col}"
                                             for col in stock_columns],
                         width=small_width)

scatter_y_format = Select(title="y-format", value="value",
                          options=["value", "absolute", "log_10(absolute)"],
                          width=small_width)

update_scatter_button = Button(label="apply changes to the scatter plot",
                               width=right_width)
update_scatter_button.on_click(update_scatter_plot)


""" Making the RadioButtonGroup for the grouping """
button_group = RadioButtonGroup(labels=grouping["keys"], width=right_width)
button_group.on_click(perform_grouping)


""" Making the widgets for the clustering """
cluster_method = Select(title="clustering method", value="k-means",
                        options=["k-means", "k-medoids", "ward's method",
                                 "complete linkage", "single linkage"],
                        width=small_width)
cluster_method.on_change("value", modify_clustering_k)
clustering_k = Slider(title="number of clusters",
                        start=1, end=7, value=3, step=1,
                        width=small_width)
cluster_attributes = Select(title="attributes",
                            value="visible on scatter plot",
                            options=["visible on scatter plot",
                                     "open, close, volume, ntweets"],
                            width=3*tiny_width+10, margin=grouping_margin)
cluster_norm = Select(title="normalize data", value="false",
                      options=["false", "true"], width=small_width)
cluster_button = Button(label="perform clustering",
                        width=fill_width, margin=button_margin)
cluster_button.on_click(perform_clustering)


""" Adding a DateRangeSlider to change the range of the data """
date_slider = DateRangeSlider(title="Date Range: ",
                             start=date_range[0], end=date_range[-1],
                             value=(date_range[0], date_range[-1]),
                             step=length_of_day, width=right_width)

date_button = Button(label="update time range",
                     width=right_width)
date_button.on_click(set_time_range)


""" Arranging the layout and showing it """
time_plots = column(plot_stock,
                    plot_tweet)

left_changes = row(column(stock_data,
                          cluster_method),
                   column(stock_ratio,
                          clustering_k),
                   column(stock_format,
                          cluster_norm),
                   column(update_upper_button,
                          cluster_attributes),
                   column(tweet_select,
                          cluster_button))

scatter_related = column(scatter_plot,
                         row(scatter_x_data,
                             scatter_x_ratio,
                             scatter_x_format),
                         row(scatter_y_data,
                             scatter_y_ratio,
                             scatter_y_format),
                         update_scatter_button)

general_changes = column(button_group,
                         date_slider,
                         date_button)

layout = row(column(time_plots,
                    left_changes),
             column(scatter_related,
                    general_changes))

layout.sizing_mode = "stretch_both"
curdoc().add_root(layout)
curdoc().title = "Interactive explorer"