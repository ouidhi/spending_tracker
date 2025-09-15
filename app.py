import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import streamlit as st
import plotly.express as px

model = joblib.load('models/logistic_bert_classifier.pkl')
bert = SentenceTransformer('all-MiniLM-L6-v2')

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

st.title("Personal Spending Analyzer")
st.write("Upload your credit card CSV file and get categorized insights!")

uploaded_file = st.file_uploader("Choose a CSV file", type= "csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.title()  # Remove spaces and standardize casing

    if all(col in df.columns for col in ['Date', 'Description', 'Amount']):
        df = preprocess(df)
        
        # BERT 
        X_input = bert.encode(df['NewDescription'])
        df['Category'] = model.predict(X_input)

        st.success("Categorization complete!")
        
        filtered_df = df[df['Category'] != 'Credit Card Payment']

        # plots

        # --- Row 1 ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Spending by Category")
            category_sums = filtered_df.groupby('Category')['Amount'].sum().reset_index()
            fig1 = px.pie(category_sums, names='Category', values='Amount')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Monthly Spending")
            monthly = filtered_df.groupby(['Year', 'Month'])['Amount'].sum().reset_index()
            monthly['Date'] = pd.to_datetime(monthly['Year'].astype(str) + '-' + monthly['Month'] + '-01')
            monthly = monthly.sort_values('Date')
            fig2 = px.line(monthly, x='Date', y='Amount')
            st.plotly_chart(fig2, use_container_width=True)


        # --- Row 2 ---
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Months Ranked by Total Spending")
            monthly_ranking = monthly.copy()
            monthly_ranking['MonthLabel'] = monthly_ranking['Month'] + ' ' + monthly_ranking['Year'].astype(str)
            monthly_ranking = monthly_ranking.sort_values(by='Amount', ascending=False)
            st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'> {' > '.join(monthly_ranking['MonthLabel'].tolist())}</h3>", unsafe_allow_html=True)

        with col4:
            st.subheader("Top 3 Spending Categories")
            top3 = filtered_df.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(3).reset_index()
            fig4 = px.bar(top3, x='Category', y='Amount', color='Category')
            st.plotly_chart(fig4, use_container_width=True)


        # --- Row 3 (full width) ---
        st.subheader("Monthly Spending Breakdown by Category")
        stacked = filtered_df.groupby(['Year', 'Month', 'Category'])['Amount'].sum().reset_index()
        stacked['Date'] = pd.to_datetime(stacked['Year'].astype(str) + '-' + stacked['Month'] + '-01')
        stacked = stacked.sort_values('Date')
        fig5 = px.bar(stacked, x='Date', y='Amount', color='Category')
        st.plotly_chart(fig5, use_container_width=True)


        # --- Row 4 (full width) ---
        st.subheader("Raw Categorized Data")
        filtered_df = filtered_df[['Month', 'Year', 'Description', 'Amount', 'Category']]
        st.dataframe(filtered_df)

    else:
        st.error("CSV must contain 'Date', 'Description' and 'Amount' columns.")





