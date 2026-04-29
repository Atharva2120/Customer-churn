# Customer Churn Prediction 📉

Predict whether a customer is likely to stop using a service (churn) using machine learning. This project includes **notebook + Python scripts**, model comparison, and feature importance insights.

## 🔗 Dataset
Kaggle: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data

> Download the dataset and place the CSV in `data/customer_churn.csv`.

## ✨ Project Highlights
- Classification models (Logistic Regression, Random Forest, Gradient Boosting)
- Feature preprocessing (imputation + one-hot encoding)
- Model evaluation (ROC-AUC, Accuracy, F1)
- Feature importance for business interpretation

## 🗂 Project Structure
```
Customer-churn/
├── data/
│   └── customer_churn.csv   # dataset (download from Kaggle)
├── notebooks/
│   └── customer_churn_prediction.ipynb
├── outputs/
│   ├── model.pkl
│   └── metrics.json
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🧪 Train Models
```bash
python src/train.py --data data/customer_churn.csv --target Churn
```

Outputs:
- `outputs/model.pkl` (best model)
- `outputs/metrics.json`

## 🔮 Predict on New Data
```bash
python src/predict.py --model outputs/model.pkl --input data/customer_churn.csv --output outputs/predictions.csv
```

## 📓 Notebook
Open the notebook for an end-to-end walkthrough:
```
notebooks/customer_churn_prediction.ipynb
```

## ✅ Recommended Model
Start with **Logistic Regression** as a baseline, then compare **Random Forest** and **Gradient Boosting**. In most tabular churn datasets, **Gradient Boosting** tends to provide the best balance of performance and interpretability.

## 👤 Author
Atharva2120
