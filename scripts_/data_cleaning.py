import pandas as pd
import numpy as np


class DataCleaning:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def percent_missing(self, df: pd.DataFrame):

        # Calculate total number of cells in dataframe
        totalCells = np.product(df.shape)

        # Count number of missing values per column
        missingCount = df.isnull().sum()

        # Calculate total number of missing values
        totalMissing = missingCount.sum()

        # Calculate percentage of missing values
        print("The Telecom dataset contains", round(
            ((totalMissing/totalCells) * 100), 2), "%", "missing values.")

        return df

    def unique_users(self, df: pd.DataFrame):

        # Check if all the users are unique
        number_of_users = df.auction_id.unique()
        print(
            f'The data has {df.shape[0] - len(number_of_users)} repeated users.')

        return df

    def list_coloumn_names(self) -> pd.DataFrame:
        return self.df.columns

    '''
    functions to get information about who responded to the ads
    '''

    def response(self, df: pd.DataFrame):

        # Users who either answered yes or no
        responeded_df = df.copy(deep=True)
        responeded_df = responeded_df[responeded_df['yes'] == 1].append(
            responeded_df[responeded_df['no'] == 1])
        # print(responeded_df.head(5))

        return responeded_df