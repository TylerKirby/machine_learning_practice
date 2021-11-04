import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import r_regression
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

np.random.seed(42)


class PearsonRTransform(BaseEstimator, TransformerMixin):
    def __init__(self, threshold, col_names, verbose=False):
        self.threshold = threshold
        self.col_names = col_names
        self.cols_to_drop = []
        self.verbose = verbose

    def fit(self, X, y):
        corr_coefs = np.abs(r_regression(X, y))
        cols_to_drop = np.where(corr_coefs <= self.threshold)[0]
        if self.verbose:
            print(f"Threshold = {self.threshold}")
            print(f"Dropping {len(cols_to_drop)} of {X.shape[1]} columns")
            print(f"Columns dropped: {self.col_names[cols_to_drop]}")
        self.cols_to_drop = cols_to_drop

    def transform(self, X, y=None):
        return np.delete(X.copy(), self.cols_to_drop, axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        X_trans = self.transform(X)
        return X_trans


if __name__ == "__main__":
    train_df = pd.read_csv("train_df.csv")
    test_df = pd.read_csv("test_df.csv")
    val_df = pd.read_csv("val_df.csv")

    X_train = train_df.drop("quality", axis=1)
    y_train = train_df["quality"]
    X_test = test_df.drop("quality", axis=1)
    y_test = test_df["quality"]
    X_val = val_df.drop("quality", axis=1)
    y_val = val_df["quality"]

    print("Linear Regression Experiment")
    linear_reg_pipeline = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("pearson_r", PearsonRTransform(0.2, X_train.columns.values, True)),
        ("linear_reg", LinearRegression())
    ], verbose=True)
    linear_reg_pipeline.fit(X_train, y_train)
    linear_reg_R2 = linear_reg_pipeline.score(X_val, y_val)
    print("Linear Regression Model R^2: ", linear_reg_R2)

    print("Support Vector Machine Experiment")
    svm_pipeline = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("pearson_r", PearsonRTransform(0.2, X_train.columns.values, True)),
        ("svm", SVR())
    ], verbose=True)
    svm_pipeline.fit(X_train, y_train)
    svm_R2 = svm_pipeline.score(X_val, y_val)
    print("Support Vector Machine Model R^2: ", svm_R2)

    print("Grid Search SVM Experiment")
    svm_pipeline = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("pearson_r", PearsonRTransform(0.2, X_train.columns.values, False)),
        ("svm", SVR())
    ], verbose=False)
    param_grid = {
        "pearson_r__threshold": [0, 0.1, 0.2],
        "svm__kernel": ["linear", "poly", "rbf"],
        "svm__C": np.linspace(0.01, 1, 10)
    }
    grid_search = GridSearchCV(svm_pipeline, param_grid)
    grid_search.fit(X_val, y_val)
    grid_search_R2 = grid_search.score(X_val, y_val)
    print(f"Best Parameters: ", grid_search.best_params_)
    print(f"Grid Search Score: ", grid_search_R2)

    print("Random Search SVM Experiment")
    param_grid = {
        "pearson_r__threshold": stats.uniform(0, 0.2),
        "svm__kernel": ["linear", "poly", "rbf"],
        "svm__C": stats.uniform(0.01, 1)
    }
    random_search = RandomizedSearchCV(svm_pipeline, param_grid, n_iter=50)
    random_search.fit(X_val, y_val)
    random_search_R2 = random_search.score(X_val, y_val)
    print(f"Best Parameters: ", random_search.best_params_)
    print(f"Random Search Score: ", random_search_R2)

    print("Final Model Results")
    svm_pipeline = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("pearson_r", PearsonRTransform(0, X_train.columns.values, False)),
        ("svm", SVR(kernel="rbf", C=0.67))
    ], verbose=False)
    svm_pipeline.fit(X_train, y_train)
    final_score = svm_pipeline.score(X_test, y_test)
    print("Final Score: ", final_score)
