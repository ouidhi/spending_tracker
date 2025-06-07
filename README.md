# 💸 Credit Card Transaction Categorizer & Budget Forecasting
(SQL + ML + Streamlit)

A data science project built using my **own real credit card transaction data** from April 2024 to April 2025. This project applies **SQL, Python, and ML** to analyze personal finance patterns, detect anomalies, and visualize spending trends using **Streamlit**.

> Predict categories from messy transaction descriptions and build monthly spending dashboards — powered by NLP, machine learning, and time-series forecasting.

---

## 🛠 Tech Stack

- **Python**: Pandas, NumPy, Matplotlib, Scikit-learn
- **SQL**: SQLite (or PostgreSQL/MySQL)
- **Streamlit**: Interactive dashboard for visualization & trend exploration
- **Jupyter Notebook**: Analysis + Modeling

---


## 📂 Data Overview

- **Time Period**: April 1, 2024 – April 1, 2025  
- **Columns**:
  - `Date`: Date of transaction
  - `Description`: Vendor or payment detail
  - `Amount`: All transactions positive; categorized manually
  - `Type`: Purchase or payment
  - `Final_Amount`: negative for spending and positive for card payment 
  - `Category`: Manually labeled (e.g., Groceries, Transit, Self-care, Payments)

> 🔐 Note: All sensitive data is anonymized for repo sharing.

---

## 🧠 Workflow

### 1. Data Cleaning
- Removed null entries and duplicates
- Converted date formats and normalized descriptions
- Made all **spending negative** and **credit card payments positive**

```python
def clean_description(desc):
    desc = desc.lower()
    desc = re.sub(r'[^a-zA-Z\s]', '', desc)
    desc = re.sub(r'\s+', ' ', desc).strip()
    return desc
```

### 2. Manual Categorization
- Used custom rules to map transaction descriptions to categories
- Added a `category` column using personal insight

### 3. Feature Engineering (Text → Vectors)

TF-IDF (Simple & Effective)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_description'])

### 4. Train the Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, df['Category'], test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

Save the model and vectorizer:

joblib.dump(model, 'models/classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

### 5. Predict New Transactions

def predict_category(text):
    cleaned = clean_description(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

    
### 6. SQL Aggregation
- Used SQL to query average spend, category-wise monthly trends, etc.
- Flagged potential outliers (e.g., unusually high daily spending)

### 7. Feature Engineering
- Created new features: transaction frequency, weekday vs. weekend spending, etc.

### 8. Forecast Future Spending
Use time series tools like Prophet or ARIMA:
Group totals by Month x Category
Predict next month's top 3 categories based on historical data

### 9. Streamlit Dashboard
- Built an interactive web app to explore:
  - Weekly/monthly spending
  - Category-wise expenses
  - Outlier alerts
  - Fraud detection model outputs

---

## 🚀 Future Enhancements

- Add more advanced fraud logic (sequence modeling, clustering)
- Support multi-account aggregation (credit + debit)
- Integrate with Plaid or YNAB API for live updates
- Add budgeting goals and alert notifications

---

## 🎯 Skills Highlighted

- Personal Finance Analysis
- Data Cleaning & Manual Categorization
- SQL + Python Integration
- Feature Engineering
- ML for Anomaly/Fraud Detection
- Streamlit for Dashboarding
- Data Storytelling from Raw Life Data

---

## 📸 Demo Screenshots (Coming Soon)

> Visuals of the dashboard, monthly breakdowns, anomaly detection, etc.

---

## 📝 License

MIT License.  
This project is for **educational and demonstrative purposes only** — not for commercial finance use.
