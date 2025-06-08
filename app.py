import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

import streamlit as st

model = joblib.load('models/logistic_bert_classifier.pkl')
bert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def clean_description(desc):
    desc = desc.lower()
    desc = re.sub(r'[^a-z]', '', desc)
    return desc

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']) 
    df['Month'] = df['Date'].dt.strftime('%b')
    df['Year'] = df['Date'].dt.year
    df['NewDescription'] = df['Description'].apply(clean_description)
    df['Amount'] = pd.to_numeric(df['Amount'])
    return df

def categorizer(desc):
    if re.search(r"(shoppers|rexall|pharmacy|drug)", desc):
        return "Pharmacy"
    elif re.search(r"(starbucks|timhortons|coffee|cafe|tea|matcha|ippodo|espressobar|espresso|varda|dean|arabica|caf|tsujiri|milkys|alley)", desc):
        return "Cafe"
    elif re.search(r"(walmart|loblaws|nofrills|grocery|metro|johndanielles|market|hmart|fortinos|subzimandi|7 eleven|convenience|eleven)", desc):
        return "Groceries"
    elif re.search(r"(ubercanadatoronto|lyft|taxi|presto|shell|petro)", desc):
        return "Transport"
    elif re.search(r"(netflix|spotify|subscription|disneyplus|bill|insurance|virginplus)", desc):
        return "Subscription"
    elif re.search(r"(payment|credit|thankyou|cashback)", desc):
        return "Credit Card Payment"
    elif re.search(r"(restaurant|dining|pizza|mcdonalds|burger|waffle|eats|chicken|kellyslanding|dairyqueen|bakery|bistro|popeyes|chatime|chipotle|brewhouse|mexicana|kitc|sweets|craveables|baskinrobbins|ice|hakkalicious|villacaledon|kfc|shanghai|turtlejacks|beavertails|food|earls|krispykreme|tacos|blueclaw|pizz|eataly|bagel|gelateria|ramen|demetres|wingstop|waffles|shakeshack|bhc|cocacola|restaur)", desc):
        return "Food & Dining"
    elif re.search(r"(dollarama|shein|hudsonsbay|yorkdale|amzn|shop|bodyworks|zara|thrift|valuevillage|kiokii|uniqlo|arden|ikea|apple|dollarstore)", desc):
        return "Shopping"
    elif re.search(r"(newnham|campus|onecard)", desc):
        return "Campus Cafeteria"
    else:
        return "Other"
    
st.title("Personal Spending Analyzer")
st.write("Upload your credit card CSV file and get categorized insights!")

uploaded_file = st.file_uploader("Chose a CSV file", type= "csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if all(col in df.columns for col in ['Date', 'Description', 'Amount']):
        df = preprocess(df)

        # BERT 
        X_input = bert.encode(df['NewDescription'])
        df['Category'] = model.predict(X_input)

        st.success("Categorization complete!")

        # plots
        st.subheader("Spending by Category")
        st.bar_chart(df.groupby('Category')['Amount'].sum())

        st.subheader("Monthly Trend")
        st.line_chart(df.groupby(['Year', 'Month'])['Amount'].sum())

        st.subheader("Raw Categorized Data")
        st.dataframe(df)

    else:
        st.error("CSV must contain 'Date', 'Description' and 'Amount' columns.")





