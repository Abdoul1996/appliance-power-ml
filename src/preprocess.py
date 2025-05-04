"""
preprocess.py - Feature engineering and selection utilities
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, f_regression

def encode_categorical(df):
    """One-hot encode categorical variables (drop first to avoid multicollinearity),
    and convert boolean columns to binary (0/1)."""
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded.astype(int)


def cap_outliers_iqr(df, column):
    """Cap outliers using IQR method for a specific column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                  np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

def extract_datetime_features(df, column='date'):
    """Extracts datetime features from a given datetime column."""
    df[column] = pd.to_datetime(df[column], errors='coerce')
    df['hour'] = df[column].dt.hour
    df['day'] = df[column].dt.day
    df['day_of_week'] = df[column].dt.dayofweek
    df['day_name'] = df[column].dt.day_name()
    df['month'] = df[column].dt.month
    df['month_name'] = df[column].dt.month_name()
    return df

def add_face_of_day(df):
    """Adds a column for face of the day (Morning, Afternoon, etc.) based on hour."""
    def face_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 24:
            return 'Evening'
        else:
            return 'Night'
    
    df['face_of_day'] = df['hour'].apply(face_of_day)
    return df

def variance_threshold_selector(X_train, X_test, threshold=0.01):
    """Removes features with low variance."""
    selector = VarianceThreshold(threshold=threshold)
    X_train_var = selector.fit_transform(X_train)
    X_test_var = selector.transform(X_test)
    
    selected_cols = X_train.columns[selector.get_support()]
    return (
        pd.DataFrame(X_train_var, columns=selected_cols),
        pd.DataFrame(X_test_var, columns=selected_cols),
        selected_cols
    )

def f_value_selector(X_train, X_test, y_train, threshold=30):
    """Selects features with F-value above a given threshold."""
    f_vals, p_vals = f_regression(X_train, y_train)
    anova_df = pd.DataFrame({
        'Feature': X_train.columns,
        'F_value': f_vals,
        'p_value': p_vals
    })
    selected_f_cols = anova_df[anova_df['F_value'] > threshold]['Feature']
    return (
        X_train[selected_f_cols],
        X_test[selected_f_cols],
        selected_f_cols,
        anova_df
    )
