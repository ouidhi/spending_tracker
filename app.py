import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import re
from sklearn.base import TransformerMixin
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class BertVectorizer(TransformerMixin):
    """BERT vectorizer for text embeddings"""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X)

class TransactionClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    def predict(self, descriptions):
        return self.model.predict(descriptions)

def clean_description(text):
    """Clean transaction description"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_file(uploaded_file):
    """Load different file formats"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or Parquet files.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def find_column(df, possible_names):
    """Find column by matching possible names (case-insensitive)"""
    df_columns_lower = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in df_columns_lower:
            return df_columns_lower[name.lower()]
    return None

def create_monthly_spending_chart(df, date_col, amount_col):
    """Create interactive monthly spending line chart"""
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col])
    df_copy['month_year'] = df_copy[date_col].dt.to_period('M').astype(str)
    
    monthly_spending = df_copy.groupby('month_year')[amount_col].sum().reset_index()
    monthly_spending = monthly_spending.sort_values('month_year')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_spending['month_year'],
        y=monthly_spending[amount_col],
        mode='lines+markers',
        name='Spending',
        line=dict(color='#a8dadc', width=3),
        marker=dict(size=10, color='#457b9d'),
        fill='tozeroy',
        fillcolor='rgba(168, 218, 220, 0.3)'
    ))
    
    fig.update_layout(
        title='💰 Monthly Spending Trend',
        xaxis_title='Month',
        yaxis_title='Amount Spent',
        template='plotly_white',
        hovermode='x unified',
        height=400,
        font=dict(family="Arial", size=12, color='#f1faee'),
        plot_bgcolor='#1d3557',
        paper_bgcolor='#1d3557',
    )
    
    return fig, monthly_spending

def create_category_chart(df, amount_col):
    """Create interactive category spending chart (exclude Income)"""
    
    plot_df = df[df['category'] != 'Income']  

    category_spending = (
        plot_df
        .groupby('category')[amount_col]
        .sum()
        .reset_index()
        .sort_values(amount_col, ascending=False)
        .head(5)
    )

    blue_colors = ['#2a6f97', '#468faf', '#61a5c2', '#89c2d9', '#a9d6e5']

    fig = px.bar(
        category_spending,
        x='category',
        y=amount_col,
        color='category',
        color_discrete_sequence=blue_colors,
        title='🏆 Top 5 Spending Categories (Excludes Income)'
    )

    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        height=400,
        xaxis_title='Category',
        yaxis_title='Total Amount',
        plot_bgcolor="#1d3557",
        paper_bgcolor='#1d3557',
        font=dict(color='#f1faee')
    )

    return fig, category_spending

def main():
    # Page config with custom theme
    st.set_page_config(
        page_title="Transaction Categorizer",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
<style>

/* App background */
[data-testid="stAppViewContainer"] {
    background-color: #1d3557;
    color: #f1faee;
}

/* Centered content width */
[data-testid="stAppViewContainer"] > .main {
    max-width: 1100px;
    margin: auto;
}

/* Headings */
h1, h2, h3 {
    color: #f1faee;
    text-align: center;
}

/* Paragraphs */
p, label, span {
    color: #f1faee;
}

/* Cards / containers */
[data-testid="stVerticalBlock"] {
    background-color: #457b9d;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;

}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #457b9d;
    border-radius: 16px;
    padding: 28px;
    border: 2px dashed #a8dadc;
}
                

/* Buttons */
button {
    background-color: #f1faee !important;
    color: #1d3557 !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    padding: 10px 28px;
    border: none;
}

button:hover {
    background-color: #ffffff !important;
    color: #1d3557 !important;
    transform: translateY(-1px);
}

/* Download button specific */
button[kind="secondary"],
button[data-testid="stDownloadButton"] button {
    background-color: #f1faee !important;
    color: #1d3557 !important;
}

button[kind="secondary"]:hover,
button[data-testid="stDownloadButton"] button:hover {
    background-color: #ffffff !important;
    color: #1d3557 !important;
}

/* Force button text color */
button p,
button span,
button div {
    color: #1d3557 !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #a8dadc;
    font-size: 30px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #457b9d;
    border-radius: 12px;
}

/* Hide sidebar completely */
[data-testid="stSidebar"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)


    # Header with emoji
    st.markdown("<h1>⊹₊˚‧Transaction Categorizer Dashboard‧˚₊⊹</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'> Get instant insights • Accurate categories • Beautiful visualizations</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div style="text-align:center;">
        <h2>Upload your transaction file (˶˃ ᵕ ˂˶)</h2>
        <p>CSV, Excel, or Parquet supported</p>
    </div>
    """, unsafe_allow_html=True)
    
    left, center, right = st.columns([1, 2, 1])

    with center:
        uploaded_file = st.file_uploader(
            "",
            type=['csv', 'xlsx', 'xls', 'parquet'],
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        df = load_file(uploaded_file)
        
        if df is not None:
            # Success message with metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Total Transactions", f"{len(df):,}")
            with col2:
                st.metric("📋 Columns", len(df.columns))
            with col3:
                st.metric("💾 File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            st.markdown("---")
            
            # Column mapping section
            st.markdown("### Map Your Columns")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                desc_suggestions = find_column(df, [
                    'description', 'transaction_description', 'desc', 
                    'merchant', 'details', 'narration', 'particulars'
                ])
                desc_col = st.selectbox(
                    "📝 Description Column*",
                    options=df.columns,
                    index=df.columns.get_loc(desc_suggestions) if desc_suggestions else 0
                )
            
            with col2:
                date_suggestions = find_column(df, [
                    'date', 'transaction_date', 'time', 'timestamp', 
                    'datetime', 'posting_date'
                ])
                date_col = st.selectbox(
                    "📅 Date Column (optional)",
                    options=['None'] + list(df.columns),
                    index=df.columns.get_loc(date_suggestions) + 1 if date_suggestions else 0
                )
            
            with col3:
                amount_suggestions = find_column(df, [
                    'amount', 'value', 'debit', 'credit', 
                    'transaction_amount', 'sum'
                ])
                amount_col = st.selectbox(
                    "💵 Amount Column (optional)",
                    options=['None'] + list(df.columns),
                    index=df.columns.get_loc(amount_suggestions) + 1 if amount_suggestions else 0
                )
            
            st.markdown("---")
            
            # Centered button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                process_button = st.button("(๑ > ᴗ < ๑)  Categorize Transactions", use_container_width=True)
            
            if process_button:
                try:
                    model_path = Path('models/logistic_bert_classifier.pkl')
                    if not model_path.exists():
                        st.error("Model file not found! (ᵕ—ᴗ—)")
                        return
                    
                    with st.spinner("🤖 AI is working its magic..."):
                        classifier = TransactionClassifier(model_path)
                        df['cleaned_description'] = df[desc_col].apply(clean_description)
                        predictions = classifier.predict(df['cleaned_description'].values)
                        df['category'] = predictions
                    
                    st.success("Categorization Complete!◟( ˃̶͈◡ ˂̶͈ )◞")
                    st.toast("Categorization complete", icon="✅")
                    
                    # Prepare result dataframe
                    columns_to_keep = ['category', desc_col]
                    if date_col != 'None':
                        columns_to_keep.insert(0, date_col)
                    if amount_col != 'None':
                        columns_to_keep.insert(1 if date_col != 'None' else 0, amount_col)
                    
                    result_df = df[columns_to_keep].copy()

                    plot_df = result_df[result_df['category'] != 'Income']
                    result_df = plot_df.copy()
                    
                    # Dashboard layout
                    if date_col != 'None' and amount_col != 'None':
                        st.markdown("## 📊 Spending Analytics Dashboard")
                        
                        # Top row: Monthly trend and top categories
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            monthly_fig, monthly_data = create_monthly_spending_chart(
                                result_df, date_col, amount_col
                            )
                            st.plotly_chart(monthly_fig, use_container_width=True)
                            
                            # Top spending months
                            st.markdown("#### 📅 Top 3 Spending Months")
                            top_months = monthly_data.nlargest(3, amount_col)
                            for idx, row in top_months.iterrows():
                                st.metric(
                                    f"🏆 {row['month_year']}", 
                                    f"${row[amount_col]:,.2f}"
                                )
                        
                        with col2:
                            category_fig, category_data = create_category_chart(
                                result_df, amount_col
                            )
                            st.plotly_chart(category_fig, use_container_width=True)
                            
                            # Top spending categories
                            st.markdown("####  Top 3 Spending Categories")
                            for idx, row in category_data.head(3).iterrows():
                                st.metric(
                                    f"💳 {row['category']}", 
                                    f"${row[amount_col]:,.2f}"
                                )
                    
                    else:
                        # Just show category distribution
                        st.markdown("## 📊 Category Distribution")
                        category_counts = result_df['category'].value_counts()
                        
                        fig = px.pie(
                            values=category_counts.values,
                            names=category_counts.index,
                            title='Transaction Categories',
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Categorized data table
                    st.markdown("---")
                    st.markdown("## 📋 Categorized Transactions")
                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download section
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Categorized Data",
                            data=csv,
                            file_name="categorized_transactions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"(ᵕ—ᴗ—) Error during categorization: {str(e)}")

if __name__ == "__main__":
    main()