import numpy as np
import pandas as pd

from train_wine_model import PearsonRTransform


train_df = pd.read_csv("train_df.csv")
X = train_df.drop("quality", axis=1)
y = train_df["quality"]


def test_PearsonRTransform_fit():
    pearson_trans = PearsonRTransform(threshold=0.2, col_names=X.columns.values, verbose=True)
    pearson_trans.fit(X, y)
    assert (pearson_trans.cols_to_drop == np.array([0,  2,  3,  5,  6,  8,  9, 11])).all()


def test_PearsonRTransform_transform():
    pearson_trans = PearsonRTransform(threshold=0.2, col_names=X.columns.values, verbose=True)
    pearson_trans.fit(X, y)
    X_trans = pearson_trans.transform(X)
    assert X_trans.shape[1] == 4
