# 🏠 House Price Prediction using XGBoost

This project uses a supervised machine learning approach to predict house prices based on various features like location, size, number of rooms, etc. The model is trained from scratch using a real-world dataset.

---

## 📁 Project Structure

```
├── house_data.csv            # Input dataset
├── train_house_price_model.ipynb  # Jupyter Notebook (Google Colab format)
├── xgb_house_price_model.pkl # Trained model
├── scaler.pkl                # Scaler object used for normalization
└── README.md                 # Project documentation
```

---

## 🔍 Overview

- **Objective:** Predict housing prices using regression.
- **Model Used:** [XGBoost Regressor](https://xgboost.readthedocs.io/)
- **Features:** Categorical & numerical real estate features.
- **Evaluation Metrics:** MAE, RMSE, R²

---

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

---

## 📊 Dataset Description

The dataset `house_data.csv` includes:

- **Numerical features:** Area, Bedrooms, Bathrooms, etc.
- **Categorical features:** Location, Type, etc.
- **Target:** `price` (continuous numeric variable)

**Note:** You can use your own dataset — just ensure the target column is named `price` or change it in the code.

---

## 🚀 Steps to Train the Model

All steps are documented in the Colab notebook:

1. Upload CSV to Colab
2. Load and inspect dataset
3. Preprocess missing values and encode categorical features
4. Train-test split
5. Feature scaling using `StandardScaler`
6. Train XGBoost Regressor
7. Evaluate model using MAE, RMSE, R²
8. Visualize predictions
9. Export model & scaler using `joblib`

---

## 📈 Sample Results

| Metric | Value   |
|--------|---------|
| MAE    | ~$3,800 |
| RMSE   | ~$5,200 |
| R²     | ~0.92   |

**Note:** Results may vary depending on the dataset.

---

## 📉 Prediction Visualization

*(Visualization details are in the notebook)*

---

## 🧠 Model Export

```python
import joblib

joblib.dump(model, 'xgb_house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

These files can be reused to deploy the model via FastAPI, Flask, or in a Streamlit dashboard.

---

## 🧪 How to Use the Model

```python
import joblib
import numpy as np

model = joblib.load("xgb_house_price_model.pkl")
 consort = joblib.load("scaler.pkl")

sample_data = np.array([[... your feature values ...]])
sample_scaled = scaler.transform(sample_data)
price_prediction = model.predict(sample_scaled)
```

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [House Price Kaggle Challenge (Ames)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## 🙌 Contributions

Pull requests are welcome. For major changes, please open an issue first.

---

## 📜 License

MIT License
