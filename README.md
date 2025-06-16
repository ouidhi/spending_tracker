# 💸 Credit Card Transaction Categorizer 

Automatically categorize your credit card transactions using NLP + Machine Learning, and visualize your spending trends in a clean, interactive Streamlit dashboard.

---
## Project Overview

This project helps users upload their personal credit card transaction data and receive:
- Automated transaction categorization using both Logistic Regression and BERT sentence embeddings.
- Visual dashboards summarizing monthly spending, top categories, and detailed breakdowns.
- A live Streamlit app that works directly with user-uploaded CSVs.

---

## 🛠 Tech Stack

- **Python**: Pandas, NumPy, Matplotlib, Scikit-learn, BERT sentence_transformers
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
  - `Category`: Manually labeled (e.g., Groceries, Transit, Shopping etc.)

---

## 🧠 Workflow

### 1. Data Preprocessing 
- Removed null entries and duplicates.
- Converted date formats and normalized descriptions.
- Categorized types as purchase and payment.

### 2. Manual Categorization
- Regex-based categorize() function assigns a label (e.g., Groceries, Transport) based on keywords in NewDescription.
- This forms the target Category column used for training/testing machine learning models.

### 3. Feature Engineering (Text → Vectors)

Text data is converted into numerical form using two NLP approaches:

- TF-IDF Vectorization: captures word importance across all transactions
- BERT Embeddings: captures semantic meaning of the transaction descriptions

### 4. Model Training and Evaluation

Trained two logistic regression classifiers and evaluated performance using classification reports and confusion matrices.

**TF-IDF + Logistic Regression**

**Accuracy: 72%**

**Weighted Avg F1-Score: 0.69**

- Transforms cleaned descriptions using TfidfVectorizer (word + bigram features) and trains a logistic regression classifier.

![image](https://github.com/user-attachments/assets/2f0141cf-2f40-4de0-9277-721612f2d149)

**BERT Embeddings + Logistic Regression**

- Uses SentenceTransformer (all-MiniLM-L6-v2) to convert descriptions into contextual sentence embeddings, followed by logistic regression.
  
![image](https://github.com/user-attachments/assets/50a1b93d-ab79-41b4-9f60-f33fbce0cc40)

**Accuracy: 87%**

**Weighted Avg F1-Score: 0.88**

The final deployed model uses BERT embeddings with logistic regression, because:
- It generalizes better to real-world transaction data, which is often noisy, or inconsistent.
- It significantly outperforms TF-IDF in terms of precision, recall, and F1-score across nearly all categories.
- It handles semantic similarity — grouping variations like “Tim Hortons,” and “TimHortons123” more effectively.


### 5. Streamlit Dashboard

Built an interactive app for users to upload their own CSV and:
- Auto-categorize transactions using the BERT model
- View spending insights via:
  - Pie chart (by category)
  - Line chart (monthly trend)
  - Ranked months by spending
  - Top 3 categories
  - Stacked bar (category breakdown per month)
  - Raw categorized transaction table

[Sample Dashboard](sample_dashboard.pdf)

### Upload your credit card statement and check out the app yourself! 

> [Click here](https://spendingtracker.streamlit.app/)


## 📬 Contact

Created by **Vidhi**  
🔗 [LinkedIn](https://www.linkedin.com/in/vidhi-parmar777/) | [Email](vidhi30th@gmail.com) 
