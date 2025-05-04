from sklearn.feature_selection import VarianceThreshold, f_regression
import pandas as pd

def variance_filter(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    selected = X.columns[selector.get_support()]
    return pd.DataFrame(X_filtered, columns=selected)

def f_score_selection(X, y, threshold=30):
    f_vals, _ = f_regression(X, y)
    return X.loc[:, f_vals > threshold]