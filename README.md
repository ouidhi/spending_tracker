# 💸 Personal Credit Card Transaction Analyzer 
(SQL + ML + Streamlit)

A data science project built using my **own real credit card transaction data** from April 2024 to April 2025. This project applies **SQL, Python, and ML** to analyze personal finance patterns, detect anomalies, and visualize spending trends using **Streamlit**.

---

## 📌 Why This Project Stands Out

- Real-world, **self-sourced dataset**: Credit card transactions from my daily life.
- Combines **SQL, ML, and Streamlit** into one cohesive pipeline.
- Focuses on **fraud detection** and **financial self-awareness**.
- Brings together **manual intuition** and **automated analysis** — no black-box categorization.

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

### 2. Manual Categorization
- Used custom rules to map transaction descriptions to categories
- Added a `category` column using personal insight

### 3. SQL Aggregation
- Used SQL to query average spend, category-wise monthly trends, etc.
- Flagged potential outliers (e.g., unusually high daily spending)

### 4. Feature Engineering
- Created new features: transaction frequency, weekday vs. weekend spending, etc.

### 5. ML Fraud Detection
- Labeled small set of transactions as “suspicious” (manual + threshold-based)
- Trained a **Random Forest Classifier** to detect anomalies
- Evaluated using confusion matrix and accuracy/recall scores

### 6. Streamlit Dashboard
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
