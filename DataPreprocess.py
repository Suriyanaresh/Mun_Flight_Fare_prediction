import pandas as pd_pandas


class DataPreprocessing:
    def train_dataprocess(self, df):

        trained_data_df = df

        trained_data_df.isnull().head()
        print(trained_data_df.isnull().sum())
        trained_data_df.dropna(inplace=True)
        trained_data_df[trained_data_df.duplicated()].head()
        trained_data_df.drop_duplicates(keep='first', inplace=True)
        print(trained_data_df)
        return trained_data_df

    def test_dataprocess(self, df):

        test_data_df = df
        test_data_df.head(15)
        test_data_df.info()
        test_data_df.describe()

        print(test_data_df.isnull().sum())
        print(test_data_df)

        return test_data_df

    def train_data_featuring(self, df):
        trained_data_df = df
        trained_data_df['Duration'] = trained_data_df['Duration'].str.replace(
            "h", '*60').str.replace(' ', '+').str.replace('m', '*1').apply(eval)
        trained_data_df["Travel_day"] = trained_data_df['Date_of_Journey'].str.split(
            '/').str[0].astype(int)
        trained_data_df["Travel_month"] = trained_data_df['Date_of_Journey'].str.split(
            '/').str[1].astype(int)
        trained_data_df.drop(["Date_of_Journey"], axis=1, inplace=True)

        trained_data_df["Departing_hour"] = pd_pandas.to_datetime(
            trained_data_df["Dep_Time"]).dt.hour
        trained_data_df["Departing_min"] = pd_pandas.to_datetime(
            trained_data_df["Dep_Time"]).dt.minute
        trained_data_df.drop(["Dep_Time"], axis=1, inplace=True)

        trained_data_df["Arriving_hour"] = pd_pandas.to_datetime(
            trained_data_df.Arrival_Time).dt.hour
        trained_data_df["Arriving_min"] = pd_pandas.to_datetime(
            trained_data_df.Arrival_Time).dt.minute
        trained_data_df.drop(["Arrival_Time"], axis=1, inplace=True)

        return trained_data_df

    def test_data_featuring(self, df):
        test_data_df = df

        test_data_df['Duration'] = test_data_df['Duration'].str.replace(
            "h", '*60').str.replace(' ', '+').str.replace('m', '*1').apply(eval)

        test_data_df["Travel_day"] = test_data_df['Date_of_Journey'].str.split(
            '/').str[0].astype(int)
        test_data_df["Travel_month"] = test_data_df['Date_of_Journey'].str.split(
            '/').str[1].astype(int)
        test_data_df.drop(["Date_of_Journey"], axis=1, inplace=True)

        test_data_df["Departing_hour"] = pd_pandas.to_datetime(
            test_data_df["Dep_Time"]).dt.hour
        test_data_df["Departing_min"] = pd_pandas.to_datetime(
            test_data_df["Dep_Time"]).dt.minute
        test_data_df.drop(["Dep_Time"], axis=1, inplace=True)

        test_data_df["Arriving_hour"] = pd_pandas.to_datetime(
            test_data_df.Arrival_Time).dt.hour
        test_data_df["Arriving_min"] = pd_pandas.to_datetime(
            test_data_df.Arrival_Time).dt.minute
        test_data_df.drop(["Arrival_Time"], axis=1, inplace=True)

        return test_data_df
