# my-statistics-project
---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository  
- **Name**: Appliances Energy Prediction Dataset  
- **Size**: 19,735 observations, 29 features (temperature, humidity, time, weather)

---

## 🧪 Methods & Workflow

1. **EDA**:
   - Distributions
   - Correlations
   - Outlier detection

2. **Preprocessing**:
   - Date parsing and feature extraction (hour, day, month)
   - Categorical encoding (`get_dummies`)
   - Outlier capping using IQR
   - Face-of-day labeling (morning, afternoon, evening, night)

3. **Feature Selection**:
   - Variance Threshold
   - F-value Selection (ANOVA)

4. **Modeling**:
   - Standardization
   - Multiple Linear Regression using `statsmodels.OLS`
   - Evaluation on test set

---

## 📈 Evaluation Metrics

| Metric | Value (Example) |
|--------|-----------------|
| RMSE   | ~92.75          |
| MAE    | ~54.45          |
| R²     | 0.14            |
| MAPE   | 66.96%          |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Abdoul1996/appliance-power-ml.git
cd appliance-power-ml

# Create a virtual environment
conda create -n env-statistic python=3.9
conda activate env-statistic

# Install requirements (to be generated separately)
pip install -r requirements.txt

# Open and run the notebooks in order
jupyter notebook
