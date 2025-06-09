import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import streamlit as st

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&display=swap');

    /* Change font + overall background */
    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
        background-color: #E6F4EA; /* matcha green */
        color: #333333;
    }

    /* Header / subheader colors */
    h1, h2, h3, .stTitle, .stSubtitle {
        color: #E96D96; /* soft pink */
    }

    /* Floating stickers */
    .sticker1 {
        position: fixed;
        top: 60px;
        right: 40px;
        z-index: 1000;
    }

    .sticker2 {
        position: fixed;
        bottom: 60px;
        left: 40px;
        z-index: 1000;
    }

    .sticker3 {
        position: fixed;
        bottom: 100px;
        right: 100px;
        z-index: 1000;
    }

    /* Optional: round corners + shadow for main content */
    .block-container {
        border-radius: 20px;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    </style>

    <!-- Floating sticker images -->
    <div class="sticker1">
        <img src="https://media.giphy.com/media/LmNwrBhejkK9EFP504/giphy.gif" width="100">
    </div>
    <div class="sticker2">
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjVmZmJkZDMxZGIyY2I5ZGRiYzEzYjQ3YjQwZGM1MzU5ZDg4MTE1YSZjdD1n/f9k1tV7HyORcngKF8v/giphy.gif" width="90">
    </div>
    <div class="sticker3">
        <img src="https://media.giphy.com/media/xUOxf48S2WvTg6rapa/giphy.gif" width="80">
    </div>
    """,
    unsafe_allow_html=True
)



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

def categorizer(desc):
    if re.search(r"(shoppers|rexall|pharmacy|drug)", desc):
        return "Pharmacy"
    elif re.search(r"(starbucks|timhortons|coffee|cafe|tea|matcha|ippodo|espressobar|espresso|varda|dean|arabica|caf|tsujiri|milkys|alley)", desc):
        return "Cafe"
    elif re.search(r"(walmart|loblaws|nofrills|grocery|metro|johndanielles|market|hmart|fortinos|subzimandi|7 eleven|convenience|eleven)", desc):
        return "Groceries"
    elif re.search(r"(ubertrip|lyft|taxi|presto|shell|petro)", desc):
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
        
        # 1 by category ------------
        st.subheader("Spending by Category")

        import plotly.express as px

        category_sums = filtered_df.groupby('Category')['Amount'].sum().reset_index()
        fig = px.pie(category_sums, names='Category', values='Amount', title='Spending by Category')
        st.plotly_chart(fig)


        # 2 by time -----------
        st.subheader("Monthly Spending")
        monthly = filtered_df.groupby(['Year', 'Month'])['Amount'].sum().reset_index()
        monthly['Date'] = pd.to_datetime(monthly['Year'].astype(str) + '-' + monthly['Month'] + '-01')
        monthly = monthly.sort_values('Date')

        fig = px.bar(monthly, x='Date', y='Amount', title='Spending by Month')
        st.plotly_chart(fig)

        # 3 top 3 
        st.subheader("Top 3 Spending Categories")
        top3 = filtered_df.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(3).reset_index()
        fig = px.bar(top3, x='Category', y='Amount', color='Category')
        st.plotly_chart(fig)

        
        # 4 by category per month
        st.subheader("Spending by Category (Each Month)")
        months = filtered_df['Month'].unique()

        for month in sorted(months, key=lambda m: pd.to_datetime(m, format='%b').month):
            month_data = filtered_df[filtered_df['Month'] == month]
            if not month_data.empty:
                st.markdown(f"### {month}")
                month_sum = month_data.groupby('Category')['Amount'].sum().reset_index()
                fig = px.pie(month_sum, names='Category', values='Amount')
                st.write("Rendering plot for", month)
                st.plotly_chart(fig)


        # dataframe
        st.subheader("Raw Categorized Data")
        filtered_df = filtered_df[['Month', 'Year', 'Description', 'Amount', 'Category']]
        st.dataframe(filtered_df) 

    else:
        st.error("CSV must contain 'Date', 'Description' and 'Amount' columns.")





