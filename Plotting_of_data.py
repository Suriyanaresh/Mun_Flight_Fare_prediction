import matplotlib.pyplot as plt_matplotlib
import seaborn as sns_seaborn
import pandas as pd_pandas


class Graph_plot:
    def cat_plots(self, df):
        trained_data_df = df

        sns_seaborn.catplot(y="Price", x="Airline", data=trained_data_df.sort_values(
            "Price", ascending=False), kind="boxen", height=9, aspect=4)

        plt_matplotlib.show()

    def violin_plots(self, df):
        trained_data_df = df
        sns_seaborn.catplot(y="Price", x="Source", data=trained_data_df.sort_values(
            "Price", ascending=False), kind="violin", height=5, aspect=4)

        plt_matplotlib.show()

    def Bar_chart(self, df):
        trained_data_df = df
        plt_matplotlib.figure(figsize=(15, 7))
        plt_matplotlib.title('Flights in that particular  month')
        ax_axis = sns_seaborn.countplot(x='Travel_month', data=trained_data_df)
        plt_matplotlib.xlabel('Month of the year')
        plt_matplotlib.ylabel('number of flights per month')
        for index_i in ax_axis.patches:
            ax_axis.annotate(int(index_i.get_height()), (index_i.get_x(
            )+0.25, index_i.get_height()+1), va='bottom', color='black')

    def number_of_airlines(self, df):

        trained_data_df = df

        plt_matplotlib.figure(figsize=(25, 8))
        plt_matplotlib.title('Number of flights per airline')
        ax_axis = sns_seaborn.countplot(x='Airline', data=trained_data_df)
        plt_matplotlib.xlabel('Airline_Name')
        plt_matplotlib.ylabel('Number_of_flights')
        plt_matplotlib.xticks(rotation=45)
        for index_i in ax_axis.patches:
            ax_axis.annotate(int(index_i.get_height()), (index_i.get_x(
            )+0.25, index_i.get_height()+1), va='bottom', color='black')

    def scatter_plot(self, df):

        trained_data_df = df

        plt_matplotlib.figure(figsize=(20, 5))
        plt_matplotlib.title('Flight_Price VS Type_of_Airlines')
        plt_matplotlib.scatter(
            trained_data_df['Airline'], trained_data_df['Price'])

        plt_matplotlib.xlabel('Type_of_Airlines')
        plt_matplotlib.ylabel('Ticket_Prices')

    def heat_map_graph(self, df):
        trained_data_df = df
        plt_matplotlib.figure(figsize=(20, 20))
        sns_seaborn.heatmap(trained_data_df.corr(), annot=True, cmap="RdYlGn")
        plt_matplotlib.show()
