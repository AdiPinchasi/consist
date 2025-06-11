import streamlit as st
import pandas as pd
import numpy as np
import os

base_path = r'C:\Adi\python_projects\consist\data\cleaned_files'
file_name = 'kpi_analysis_output.csv'
file_path = os.path.join(base_path, file_name)

# Read the CSV
df = pd.read_csv(file_path)

# Load your dataframe (replace with actual data source)
df = pd.read_csv(file_path, parse_dates=['invoice_date', 'due_date'])

# KPIs
match_rate = df['match_flag'].mean()
approval_rate = df['invoice_approved'].mean()
avg_similarity = df['similarity_score'].mean()
mean_abs_error = df['abs_error'].mean()
mean_rel_error = df['relative_error_pct'].mean()
avg_days_to_due = df['days_to_due'].mean()

st.title("Procurement Matching Dashboard")

# Display KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Match Rate", f"{match_rate:.1%}")
col2.metric("Approval Rate", f"{approval_rate:.1%}")
col3.metric("Avg Description Similarity", f"{avg_similarity:.2f}")

st.metric("Mean Absolute Error", f"{mean_abs_error:.2f}")
st.metric("Mean Relative Error (%)", f"{mean_rel_error:.2f}")
st.metric("Avg Days to Due", f"{avg_days_to_due:.1f}")

# Payment Status Breakdown
payment_counts = df['payment_status'].value_counts()
st.bar_chart(payment_counts)

# Correspondence Bin Distribution
corr_bin_counts = df['correspondence_bin'].value_counts()
st.bar_chart(corr_bin_counts)

# Filters
vendor_filter = st.selectbox("Select Vendor ID", options=['All'] + list(df['vendor_id'].unique()))
if vendor_filter != 'All':
    filtered_df = df[df['vendor_id'] == vendor_filter]
else:
    filtered_df = df

# Display filtered table
st.dataframe(filtered_df[['invoice_number', 'po_number', 'total_amount', 'payment_status', 'match_flag', 'similarity_score', 'invoice_approved']])

# Recommendations
st.header("Recommendations")
st.markdown("""
- **Improve matching** by tuning similarity thresholds and error tolerances.
- **Automate approval** for invoices with similarity > 0.85 and relative error < 5%.
- **Monitor KPIs** regularly to catch process degradation.
- **Flag invoices** with low similarity or high error for manual review.
""")
