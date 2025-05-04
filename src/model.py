import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_ols(X_train, y_train):
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const)
    return model.fit()

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

def predict_ols(model, X):
    """Add constant and predict using trained OLS model."""
    X_const = sm.add_constant(X, has_constant='add')
    return model.predict(X_const)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Full OLS pipeline: train, predict, evaluate."""
    model = train_ols(X_train, y_train)
    y_pred = predict_ols(model, X_test)
    metrics = evaluate_model(y_test, y_pred)
    return model, y_pred, metrics

def print_metrics(metrics_dict):
    print("\nEvaluation Metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.2f}")