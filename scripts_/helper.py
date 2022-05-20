import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, MinMaxScaler
import pickle

from app_logger import App_Logger


class Helper:

  def __init__(self):
      self.logger = App_Logger("helper.log").get_app_logger()

  def read_model(self, file_name):
    with open(f"../models/{file_name}.pkl", "rb") as f:
        self.logger.info(f"Model loaded from {file_name}.pkl")
        return pickle.load(f)

  def write_model(self, file_name, model):
      with open(f"../models/{file_name}.pkl", "wb") as f:
          self.logger.info(f"Model dumped to {file_name}.pkl")
          pickle.dump(model, f)

  def read_csv(self, csv_path, missing_values=[]):
    try:
        df = pd.read_csv(csv_path, na_values=missing_values)
        print("file read as csv")
        self.logger.info(f"file read as csv from {csv_path}")
        return df
    except FileNotFoundError:
        print("file not found")
        self.logger.error(f"file not found, path:{csv_path}")

  def save_csv(self, df, csv_path):
    try:
        df.to_csv(csv_path, index=False)
        print('File Successfully Saved.!!!')
        self.logger.info(f"File Successfully Saved to {csv_path}")

    except Exception:
        print("Save failed...")
        self.logger.error(f"saving failed")

    return df

  def percent_missing(self, df: pd.DataFrame) -> float:

    totalCells = np.product(df.shape)
    missingCount = df.isnull().sum()
    totalMissing = missingCount.sum()
    return round((totalMissing / totalCells) * 100, 2)

  def percent_missing_for_col(self, df: pd.DataFrame, col_name: str) -> float:
    total_count = len(df[col_name])
    if total_count <= 0:
        return 0.0
    missing_count = df[col_name].isnull().sum()

    return round((missing_count / total_count) * 100, 2)

  def normalizer(self, df, columns):
    norm = Normalizer()
    return pd.DataFrame(norm.fit_transform(df), columns=columns)

  def scaler(self, df, columns):
    minmax_scaler = MinMaxScaler()
    return pd.DataFrame(minmax_scaler.fit_transform(df), columns=columns)

  def scale_and_normalize(self, df, columns):
    return self.normalizer(self.scaler(df, columns), columns)
