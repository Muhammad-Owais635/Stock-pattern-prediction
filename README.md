# 📈 Stock Pattern Prediction using Random Forest

This project uses a Random Forest Classifier to predict candlestick patterns in 15-minute stock data intervals. It demonstrates the full workflow from preprocessing, training, and evaluation to saving the trained model.

---

## 🧠 Features

- Preprocesses financial time series data
- Trains a Random Forest model to classify candlestick patterns
- Evaluates model accuracy and generates classification report
- Saves trained model using `joblib` for future inference

---

## 📁 Dataset

- Input file: `stock_data_15m.csv`
- Required columns:
  - Feature columns (e.g. OHLC values)
  - `pattern`: Target candlestick pattern label
  - `interval` and `index` columns are dropped during preprocessing

---

## 🛠️ Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
