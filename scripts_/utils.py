import pandas as pd
import dvc.api

# To Split our train data
from sklearn.model_selection import train_test_split

# To Preproccesing our data
from sklearn.preprocessing import LabelEncoder

class Utils:
    def load_data_dvc(self,tag:str, data_path: str, repo:str) -> pd.DataFrame:
        """
        Load data from a csv file.
        """
        with dvc.api.open(
            repo=repo, 
            path=data_path, 
            rev=tag,
            mode="r"
        ) as fd:
            df = pd.read_csv(fd)
        return df

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from a csv file.
        """
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print("File not found.")
        return df

    def save_csv(self, df:pd.DataFrame, csv_path:str):
        """
        Save data to a csv file.
        """
        df.to_csv(csv_path, index=False)
        try:
            df.to_csv(csv_path, index=False)
            print('File Successfully Saved.!!!')
        except Exception:
            print("Save failed...")
        return df

    def split_train_test_val(X:pd.DataFrame, Y:pd.DataFrame, size:tuple)-> list:
        """
        Split the data into train, test and validation.
        """
        train_x, temp_x, train_y, temp_y = train_test_split(X, Y, train_size=size[0], test_size=size[1]+size[2], random_state=42)
        test_x, val_x, test_y, val_y = train_test_split(temp_x, temp_y, train_size=size[1]/(size[1]+size[2]), test_size=size[2]/(size[1]+size[2]), random_state=42)
        return train_x, train_y, test_x, test_y, val_x, val_y

    def encode_features(df:pd.DataFrame)-> pd.DataFrame:
        """
        Encode features using LabelEncoder.
        """
        features = df.columns
        for feature in features:
            le = LabelEncoder()
            le.fit(df[feature])
            df[feature] = le.transform(df[feature])
        return df
        