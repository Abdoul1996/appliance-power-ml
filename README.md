# 🧠 Appliance Energy Consumption Prediction

A machine learning project for analyzing and predicting household appliance energy consumption using multiple regression models and tree-based algorithms.

---

## 📂 Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)  
- **Dataset**: Appliances Energy Prediction  
- **Size**: 19,735 observations × 29 features  
- **Features**: Includes temperature, humidity, date/time, weather conditions, etc.

---

## 🔍 Workflow

### 1. **Exploratory Data Analysis (EDA)**
- Visualized distributions and feature correlations
- Detected and handled outliers
- Analyzed time-based consumption trends

### 2. **Preprocessing**
- Extracted date/time features (hour, day, month)
- Performed categorical encoding using `get_dummies`
- Applied IQR capping to treat outliers
- Labeled time-of-day categories (morning, afternoon, evening, night)

### 3. **Feature Selection**
- Removed low-variance features
- Selected features using ANOVA F-values

### 4. **Modeling & Evaluation**
- Applied standardization
- Trained and evaluated the following models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Grid Search-optimized Decision Tree
  - Random Forest Regressor

---

## 📊 Model Comparison

| Model             | RMSE   | MAE   | R²   | MAPE (%) |
|------------------|--------|-------|------|----------|
| Linear Regression | 92.75  | 54.45 | 0.14 | 66.96    |
| Ridge Regression  | 92.75  | 54.45 | 0.14 | 66.95    |
| Lasso Regression  | 92.74  | 54.44 | 0.14 | 66.95    |
| Decision Tree     | 90.72  | 50.00 | 0.18 | 58.58    |
| Grid Search Tree  | 85.04  | 37.94 | 0.28 | 35.65    |
| **Random Forest** | **63.10**  | **29.55** | **0.60** | **29.54** |

✅ **Best Model**: `Random Forest` — achieved the highest R² score and lowest error metrics.
---

## 🔮 Forecasting

In addition to traditional regression modeling, the project includes time series forecasting to predict future appliance energy usage. Steps include:

- Resampling and aggregating timestamped data
- Visualizing seasonal and daily patterns
- Implementing forecasting models such as:
  - Moving Average
  - Exponential Smoothing
  - ARIMA (AutoRegressive Integrated Moving Average)
- Evaluating forecasts with time-based cross-validation and performance metrics (e.g., MAE, RMSE)

These forecasting models help anticipate future consumption trends and support energy optimization decisions.
