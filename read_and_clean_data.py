import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import KMeans
from functools import reduce
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import combinations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from rapidfuzz import fuzz


# Set your directory containing .txt files
current_directory = os.getcwd()
folder_path = current_directory + os.sep +'data'
# Use raw string for Windows paths
output_folder = os.path.join(folder_path, 'cleaned_files')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List of known or common date column names to try parsing
DATE_COLUMNS = ['gr_date', 'invoice_date', 'po_date', 'approval_date', 'date']

# Loop through all .txt files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        try:
            # Read the txt file as CSV
            df = pd.read_csv(file_path, skipinitialspace=True)

            # Clean column names
            df.columns = df.columns.str.strip()

            # Strip whitespace in string cells
            df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

            # Convert known date columns if they exist
            for col in df.columns:
                if col.lower() in DATE_COLUMNS:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # Save cleaned version
            output_file = os.path.join(output_folder, filename.replace('.txt', '_cleaned.csv'))
            df.to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f" Cleaned: {filename} → {output_file}")

        except Exception as e:
            print(f" Failed to process {filename}: {e}")




#data quality analysis and anomaly detection on goods_receipts_cleaned.csv

# Load cleaned goods_receipts file
file_path = os.path.join(folder_path, 'cleaned_files','goods_receipts_cleaned.csv')
gr = pd.read_csv(file_path, encoding='utf-8-sig')

# 1. Basic info
print("\n--- Data Summary ---")
print(gr.info())
print(gr.describe(include='all'))

# 2. Check for missing values
print("\n--- Missing Values ---")
print(gr.isnull().sum())

# 3. Check for duplicate GR numbers
duplicate_gr = gr[gr.duplicated(subset='gr_number', keep=False)]
print(f"\n--- Duplicate GR Numbers ---\nTotal duplicates: {len(duplicate_gr)}")
if not duplicate_gr.empty:
    print(duplicate_gr.sort_values(by='gr_number'))

# 4. Validate data types
# Convert gr_date to datetime
gr['gr_date'] = pd.to_datetime(gr['gr_date'], errors='coerce')
invalid_dates = gr[gr['gr_date'].isnull()]
print(f"\n--- Invalid GR Dates ---\nRows with unparseable 'gr_date': {len(invalid_dates)}")

# 5. Check for non-positive or missing received_quantity
invalid_qty = gr[(gr['received_quantity'].isnull()) | (gr['received_quantity'] <= 0)]
print(f"\n--- Invalid Quantities ---\nRows with missing or non-positive 'received_quantity': {len(invalid_qty)}")

# 6. Outlier detection on received_quantity (top 1%)
if 'received_quantity' in gr.columns:
    q99 = gr['received_quantity'].quantile(0.99)
    outliers = gr[gr['received_quantity'] > q99]
    print(f"\n--- Quantity Outliers ---\nValues above 99th percentile ({q99:.2f}): {len(outliers)}")
    if not outliers.empty:
        print(outliers[['gr_number', 'received_quantity', 'gr_date']].sort_values(by='received_quantity', ascending=False))

# 7. Check GR distribution by date (optional anomaly insight)
gr_per_day = gr.groupby(gr['gr_date'].dt.date).size()
print("\n--- GR Count by Date ---")
print(gr_per_day.sort_values(ascending=False).head())




#data quality analysis and anomaly detection on  purchase_orders_cleaned.csv
# Load cleaned purchase_orders file
file_path = os.path.join(folder_path, 'cleaned_files','purchase_orders_cleaned.csv')

po = pd.read_csv(file_path, encoding='utf-8-sig')

# 1. Basic info
print("\n--- Purchase Orders Data Summary ---")
print(po.info())
print(po.describe(include='all'))

# 2. Missing values check
print("\n--- Missing Values ---")
print(po.isnull().sum())

# 3. Duplicate PO numbers and line items (composite key)
duplicate_po = po[po.duplicated(subset=['po_number', 'po_line_item'], keep=False)]
print(f"\n--- Duplicate PO Number + Line Items ---\nTotal duplicates: {len(duplicate_po)}")
if not duplicate_po.empty:
    print(duplicate_po.sort_values(by=['po_number', 'po_line_item']))

# 4. Convert dates to datetime and check invalid dates
po['order_date'] = pd.to_datetime(po['order_date'], errors='coerce')
po['delivery_date'] = pd.to_datetime(po['delivery_date'], errors='coerce')

invalid_order_dates = po[po['order_date'].isnull()]
invalid_delivery_dates = po[po['delivery_date'].isnull()]

print(f"\n--- Invalid Order Dates ---\nRows with unparseable 'order_date': {len(invalid_order_dates)}")
print(f"\n--- Invalid Delivery Dates ---\nRows with unparseable 'delivery_date': {len(invalid_delivery_dates)}")

# 5. Check for logical inconsistencies (delivery_date before order_date)
invalid_date_logic = po[po['delivery_date'] < po['order_date']]
print(f"\n--- Delivery Date before Order Date ---\nRows: {len(invalid_date_logic)}")

# 6. Check for non-positive quantities, prices, or total amounts
invalid_qty = po[(po['quantity'].isnull()) | (po['quantity'] <= 0)]
invalid_unit_price = po[(po['unit_price'].isnull()) | (po['unit_price'] <= 0)]
invalid_total_amount = po[(po['total_amount'].isnull()) | (po['total_amount'] <= 0)]

print(f"\n--- Invalid Quantities (<=0 or missing) --- Rows: {len(invalid_qty)}")
print(f"--- Invalid Unit Prices (<=0 or missing) --- Rows: {len(invalid_unit_price)}")
print(f"--- Invalid Total Amounts (<=0 or missing) --- Rows: {len(invalid_total_amount)}")

# 7. Check if total_amount matches quantity * unit_price (tolerance for floating point)
po['calc_total'] = po['quantity'] * po['unit_price']
mismatch_total = po[abs(po['total_amount'] - po['calc_total']) > 0.01]
print(f"\n--- Total Amount Mismatch with Quantity * Unit Price --- Rows: {len(mismatch_total)}")

# 8. Outlier detection on quantity (top 1%)
q99_qty = po['quantity'].quantile(0.99)
qty_outliers = po[po['quantity'] > q99_qty]
print(f"\n--- Quantity Outliers (above 99th percentile = {q99_qty:.2f}) --- Rows: {len(qty_outliers)}")

# 9. Outlier detection on unit price (top 1%)
q99_price = po['unit_price'].quantile(0.99)
price_outliers = po[po['unit_price'] > q99_price]
print(f"\n--- Unit Price Outliers (above 99th percentile = {q99_price:.2f}) --- Rows: {len(price_outliers)}")




# Load vendor_invoices cleaned file
file_path = os.path.join(folder_path, 'cleaned_files','vendor_invoices_cleaned.csv')

vi = pd.read_csv(file_path, encoding='utf-8-sig')

print("\n--- Vendor Invoices Data Summary ---")
print(vi.info())
print(vi.describe(include='all'))

# 1. Missing values check
print("\n--- Missing Values ---")
print(vi.isnull().sum())

# 2. Duplicate invoice_number + po_number + po_line_item (composite key)
duplicate_invoices = vi[vi.duplicated(subset=['invoice_number', 'po_number', 'po_line_item'], keep=False)]
print(f"\n--- Duplicate Invoice + PO + Line Items ---\nTotal duplicates: {len(duplicate_invoices)}")
if not duplicate_invoices.empty:
    print(duplicate_invoices.sort_values(by=['invoice_number', 'po_number', 'po_line_item']))

# 3. Check for non-positive or missing quantities, unit prices, total amounts
invalid_qty = vi[(vi['quantity'].isnull()) | (vi['quantity'] <= 0)]
invalid_unit_price = vi[(vi['unit_price'].isnull()) | (vi['unit_price'] <= 0)]
invalid_total_amount = vi[(vi['total_amount'].isnull()) | (vi['total_amount'] <= 0)]

print(f"\n--- Invalid Quantities (<=0 or missing) --- Rows: {len(invalid_qty)}")
print(f"--- Invalid Unit Prices (<=0 or missing) --- Rows: {len(invalid_unit_price)}")
print(f"--- Invalid Total Amounts (<=0 or missing) --- Rows: {len(invalid_total_amount)}")

# 4. Check if total_amount matches quantity * unit_price (allowing small floating point tolerance)
vi['calc_total'] = vi['quantity'] * vi['unit_price']
mismatch_total = vi[abs(vi['total_amount'] - vi['calc_total']) > 0.01]
print(f"\n--- Total Amount Mismatch with Quantity * Unit Price --- Rows: {len(mismatch_total)}")

# 5. Outlier detection on quantity (top 1%)
q99_qty = vi['quantity'].quantile(0.99)
qty_outliers = vi[vi['quantity'] > q99_qty]
print(f"\n--- Quantity Outliers (above 99th percentile = {q99_qty:.2f}) --- Rows: {len(qty_outliers)}")

# 6. Outlier detection on unit price (top 1%)
q99_price = vi['unit_price'].quantile(0.99)
price_outliers = vi[vi['unit_price'] > q99_price]
print(f"\n--- Unit Price Outliers (above 99th percentile = {q99_price:.2f}) --- Rows: {len(price_outliers)}")





#Vizuasation
if 1:
    import pandas as pd
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set visual style
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)

    # Path setup
    base_path =os.path.join(folder_path,'cleaned_files')

    files = {
        'Goods Receipts': 'goods_receipts_cleaned.csv',
        'Purchase Orders': 'purchase_orders_cleaned.csv',
        'Vendor Invoices': 'vendor_invoices_cleaned.csv'
    }

    for title, fname in files.items():
        print(f"\n📂 Processing: {title}")
        path = os.path.join(base_path, fname)
        df = pd.read_csv(path, encoding='utf-8-sig')

        # Try converting date columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Missing values heatmap
        plt.figure()
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title(f"Missing Values Heatmap - {title}")
        plt.show()

        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            plt.figure()
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f"Distribution of {col} - {title}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

        # Boxplot for numeric columns to detect outliers
        for col in numeric_cols:
            plt.figure()
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col} - {title}")
            plt.show()

        # Bar plot of top 10 categories (categorical columns)
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            plt.figure()
            top_values = df[col].value_counts().nlargest(10)
            sns.barplot(x=top_values.values, y=top_values.index, palette="magma")
            plt.title(f"Top 10 values in {col} - {title}")
            plt.xlabel("Count")
            plt.ylabel(col)
            plt.show()

        # Time-based trends (if any date column exists)
        date_cols = df.select_dtypes(include='datetime64[ns]').columns
        for col in date_cols:
            plt.figure()
            time_series = df[col].dropna().dt.to_period("M").value_counts().sort_index()
            time_series.plot(kind='bar')
            plt.title(f"Records by Month - {col} ({title})")
            plt.ylabel("Count")
            plt.xlabel("Month")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()




# Path to CSV
base_path =os.path.join(folder_path,'cleaned_files')

file_path = os.path.join(base_path, 'invoice_approvals_cleaned.csv')

# Read CSV
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Drop rows with missing approver or approval status
df = df.dropna(subset=['approver', 'approval_status'])

# Normalize approval_status values (e.g., lower case)
df['approval_status'] = df['approval_status'].str.lower().str.strip()

# Count approvals by approver
approval_summary = df.groupby('approver')['approval_status'].value_counts().unstack(fill_value=0)

# Add totals and approval rate
approval_summary['total'] = approval_summary.sum(axis=1)
approval_summary['approval_rate'] = approval_summary.get('approved', 0) / approval_summary['total']

# Sort by approval rate descending
approval_summary_sorted = approval_summary.sort_values(by='approval_rate', ascending=False)

# Show result
print(approval_summary_sorted[['approved', 'total', 'approval_rate']])



#Identify trends in approvals by item types

# Define file path
base_path =os.path.join(folder_path,'cleaned_files')
file_name = 'vendor_invoices_cleaned.csv'
file_path = os.path.join(base_path, file_name)

# Read the CSV file
df = pd.read_csv(file_path)

# Use vendor_material_description as item type
df['item_type'] = df['vendor_material_description'].str.strip()

# Normalize payment_status (e.g., assume 'Paid' means approved)
df['payment_status'] = df['payment_status'].str.strip().str.lower()

# Group by item type and payment status
item_summary = df.groupby(['item_type', 'payment_status']).size().unstack(fill_value=0)

# Add total and approval rate
item_summary['Total'] = item_summary.sum(axis=1)
item_summary['Approval_Rate'] = item_summary.get('paid', 0) / item_summary['Total']

# Sort by approval rate
top_items = item_summary.sort_values(by='Approval_Rate', ascending=False)

print("\nItem types sorted by approval rate:\n")
print(top_items[['Total', 'Approval_Rate']])

# Optional: Bar plot
top_items['Approval_Rate'].plot(kind='barh', figsize=(10, 6), color='skyblue')
plt.xlabel('Approval Rate')
plt.ylabel('Item Type')
plt.title('Item Types by Approval Rate')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Examine the average times for invoice approval



# Define the file path
base_path =os.path.join(folder_path,'cleaned_files')
file_name = 'invoice_approvals_cleaned.csv'
file_path = os.path.join(base_path, file_name)

# Load the CSV
df = pd.read_csv(file_path)

# Ensure correct data types
df['processing_time_hours'] = pd.to_numeric(df['processing_time_hours'], errors='coerce')
df['approval_date'] = pd.to_datetime(df['approval_date'], errors='coerce')

# Drop rows with missing processing time
df_clean = df.dropna(subset=['processing_time_hours'])

# 1. Overall average approval time
overall_avg = df_clean['processing_time_hours'].mean()
print(f"\n Overall average approval time: {overall_avg:.2f} hours")

# 2. Average approval time by approval level
by_level = df_clean.groupby('approval_level')['processing_time_hours'].mean().sort_values()
print("\n  Average approval time by approval level:\n")
print(by_level)

# 3. (Optional) Average approval time by approver
by_approver = df_clean.groupby('approver')['processing_time_hours'].mean().sort_values()
print("\n Average approval time by approver:\n")
print(by_approver)

# 4. (Optional) Trend over time (e.g., by month)
df_clean = df_clean.copy()  # Ensure it's a full copy, not a view
df_clean['approval_month'] = df_clean['approval_date'].dt.to_period('M')

monthly_avg = df_clean.groupby('approval_month')['processing_time_hours'].mean()

# Plot monthly trend


monthly_avg.plot(kind='line', marker='o', figsize=(10,5), title='Average Invoice Approval Time Over Time')
plt.ylabel('Avg Processing Time (Hours)')
plt.xlabel('Approval Month')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



#Identify differences between different departments or purchasing coordinators



# Define file path
base_path =os.path.join(folder_path,'cleaned_files')
file_name = 'invoice_approvals_cleaned.csv'
file_path = os.path.join(base_path, file_name)

# Read the CSV
df = pd.read_csv(file_path)

# Clean up column values
df['approver'] = df['approver'].astype(str).str.strip()
df['approval_status'] = df['approval_status'].astype(str).str.strip().str.lower()
df['processing_time_hours'] = pd.to_numeric(df['processing_time_hours'], errors='coerce')

# Drop rows with missing approver or processing time
df_clean = df.dropna(subset=['approver', 'processing_time_hours']).copy()

# Group by approver
summary = df_clean.groupby('approver').agg(
    Total_Approvals=('approval_id', 'count'),
    Approved_Count=('approval_status', lambda x: (x == 'approved').sum()),
    Rejected_Count=('approval_status', lambda x: (x == 'rejected').sum()),
    Avg_Processing_Hours=('processing_time_hours', 'mean'),
    Levels_Used=('approval_level', pd.Series.nunique)
)

# Add approval rate
summary['Approval_Rate'] = summary['Approved_Count'] / summary['Total_Approvals']

# Sort by Approval Rate
summary_sorted = summary.sort_values(by='Approval_Rate', ascending=False)

print("\n Purchasing Coordinator Summary:\n")
print(summary_sorted)

# Optional: Plot average processing time
summary_sorted['Avg_Processing_Hours'].plot(kind='barh', figsize=(10,6), color='skyblue')
plt.title('Avg Processing Time by Approver')
plt.xlabel('Hours')
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()



#

# Define file path
base_path =os.path.join(folder_path,'cleaned_files')
file_name = 'vendor_invoices_cleaned.csv'
file_path = os.path.join(base_path, file_name)

# Read the CSV
df = pd.read_csv(file_path)


#clean it again


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text

df["description_cleaned"] = df["vendor_material_description"].apply(clean_text)

#vectorize description



vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["description_cleaned"])

#Create Textual Similarity Features
#Pairwise Cosine Similarity


similarity_matrix = cosine_similarity(tfidf_matrix)
#This gives a matrix where similarity_matrix[i, j] is the similarity between row i and row j

#Create a new feature — average similarity of each description to others:


df["avg_desc_similarity"] = similarity_matrix.mean(axis=1)

#clustering Similar Descriptions


n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
df["description_cluster"] = kmeans.fit_predict(tfidf_matrix)


# summary
# avg_desc_similarity
# description_cluster

# These can now be used as features in your model.



#Calculate indices of correspondence between quantities and prices
# Ensure numeric columns are properly converted
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')

# Recalculate expected total
df['recalculated_total'] = df['quantity'] * df['unit_price']

# Absolute error
df['abs_error'] = (df['recalculated_total'] - df['total_amount']).abs()

# Relative error (% difference)
df['relative_error_pct'] = (
    df['abs_error'] / df['total_amount'].replace(0, pd.NA)
).round(4) * 100

# Boolean flag if the recalculated total matches the recorded total (within small tolerance)
df['match_flag'] = df['abs_error'] < 1e-2  # tolerance of 0.01 currency units

# Optional: index of correspondence (inverse of relative error)
df['correspondence_index'] = 1 / (1 + df['relative_error_pct'].fillna(0))


#Generate supplier-based historical features from your vendor_invoice_cleaned.csv, grouped by vendor_id.

#Basic Supplier Stats
supplier_stats = df.groupby('vendor_id').agg(
    num_invoices=('invoice_number', 'nunique'),
    total_quantity=('quantity', 'sum'),
    total_spend=('total_amount', 'sum'),
    avg_unit_price=('unit_price', 'mean'),
    avg_total_amount=('total_amount', 'mean'),
    median_total_amount=('total_amount', 'median')
).reset_index()

#Ensure invoice_date is datetime:
df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')

#compute
supplier_time = df.groupby('vendor_id').agg(
    first_invoice_date=('invoice_date', 'min'),
    last_invoice_date=('invoice_date', 'max'),
    avg_invoice_interval_days=('invoice_date', lambda x: x.sort_values().diff().dt.days.mean())
).reset_index()

#Payment Behavior:
df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
df['days_to_due'] = (df['due_date'] - df['invoice_date']).dt.days

supplier_payment = df.groupby('vendor_id').agg(
    avg_days_to_due=('days_to_due', 'mean'),
    num_paid=('payment_status', lambda x: (x == 'Paid').sum()),
    num_unpaid=('payment_status', lambda x: (x != 'Paid').sum()),
    pct_paid=('payment_status', lambda x: (x == 'Paid').mean())
).reset_index()


#item diversity
supplier_items = df.groupby('vendor_id').agg(
    unique_items=('vendor_material_description', 'nunique'),
    avg_quantity_per_item=('quantity', 'mean')
).reset_index()

#final merge


dfs = [supplier_stats, supplier_time, supplier_payment, supplier_items]
supplier_history = reduce(lambda left, right: pd.merge(left, right, on='vendor_id', how='outer'), dfs)

#basic model

df['invoice_approved'] = df['payment_status'].apply(lambda x: 1 if x.lower() == 'paid' else 0)

# proces data


# Select features
features = [
    'quantity', 'unit_price', 'total_amount', 'days_to_due',
    'correspondence_index', 'currency', 'vendor_material_description'
]
X = df[features]
y = df['invoice_approved']


#build pipe line



# Preprocessing pipelines
numeric_features = ['quantity', 'unit_price', 'total_amount', 'days_to_due', 'correspondence_index']
categorical_features = ['currency']
text_feature = 'vendor_material_description'

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('txt', TfidfVectorizer(max_features=50), text_feature)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))




# explanation
#So cross-validation is fully integrated inside GridSearchCV — Grid Search with Cross-Validation

# Cross-validation is the method used to evaluate model performance reliably.
#
# Hyperparameter tuning is done by searching through all combinations in param_grid.
#
# Both are done together and automatically inside GridSearchCV.fit().

# Defining columns
numeric_features = ['quantity', 'unit_price', 'total_amount', 'days_to_due', 'correspondence_index']
categorical_features = ['currency']
text_feature = 'vendor_material_description'

# Handling missing values ​​and types
df = df.dropna(subset=['vendor_material_description', 'payment_status'])  # הסרה של שורות עם ערכים חסרים בטקסט
df['invoice_approved'] = df['payment_status'].apply(lambda x: 1 if x.lower() == 'paid' else 0)

# Characteristics and goal column
X = df[numeric_features + categorical_features + [text_feature]]
y = df['invoice_approved']

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('txt', TfidfVectorizer(max_features=100), text_feature)
])

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

# Grid Search: Hyperparameter tuning
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit with Cross-validation
grid_search.fit(X, y)

# th bst results
print("Best Parameters:", grid_search.best_params_)
print("\nBest CV Score (F1):", grid_search.best_score_)

# examine the performance on the test set
best_model = grid_search.best_estimator_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))

# Generate report as a dict
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Optional: round values for better readability
report_df = report_df.round(3)

print(report_df)
#So cross-validation is fully integrated inside GridSearchCV — Grid Search with Cross-Validation

# Cross-validation is the method used to evaluate model performance reliably.
#
# Hyperparameter tuning is done by searching through all combinations in param_grid.
#
# Both are done together and automatically inside GridSearchCV.fit().



# explain tconfusion maix and FP and FN


# Assuming y_test and y_pred are defined after model prediction
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rejected', 'Approved'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()




def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df['desc_clean'] = df['vendor_material_description'].apply(clean_text)
pairs = list(combinations(df['desc_clean'].unique(), 2))



def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text





tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['desc_clean'])

similarity = cosine_similarity(tfidf_matrix)


df['similarity_score'] = df['desc_clean'].apply(lambda x: fuzz.token_sort_ratio(x, "reference item"))



# Optional: select first 20 items for readability
n = 20
sim_subset = similarity[:n, :n]
labels = df['desc_clean'].iloc[:n]

plt.figure(figsize=(12, 10))
sns.heatmap(sim_subset, xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=False)
plt.title('Cosine Similarity Between Descriptions')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


#. Approval Rate by Key KPIs


# Add approval column if not already there
df['invoice_approved'] = df['payment_status'].str.lower().eq('paid').astype(int)

# Group by vendor
approval_by_vendor = df.groupby('vendor_id')['invoice_approved'].agg(['mean', 'count']).reset_index().rename(
    columns={'mean': 'approval_rate', 'count': 'invoice_count'}
)

# Group by material description
approval_by_item = df.groupby('vendor_material_description')['invoice_approved'].agg(['mean', 'count']).reset_index().rename(
    columns={'mean': 'approval_rate', 'count': 'invoice_count'}
)

# Group by description_cluster
approval_by_cluster = df.groupby('description_cluster')['invoice_approved'].agg(['mean', 'count']).reset_index().rename(
    columns={'mean': 'approval_rate', 'count': 'invoice_count'}
)

# Example: approval by correspondence index bucket
df['correspondence_bin'] = pd.cut(df['correspondence_index'], bins=[0, 0.5, 0.8, 1.1], labels=['Low', 'Medium', 'High'])
approval_by_corr = df.groupby('correspondence_bin')['invoice_approved'].agg(['mean', 'count']).reset_index()


#Identify Outlier Suppliers or Items

# Flag outlier vendors with low approval rates and at least 10 invoices
outlier_vendors = approval_by_vendor[
    (approval_by_vendor['approval_rate'] < 0.3) & (approval_by_vendor['invoice_count'] >= 10)
]

# Flag outlier items
outlier_items = approval_by_item[
    (approval_by_item['approval_rate'] < 0.3) & (approval_by_item['invoice_count'] >= 10)
]


#Analyze Relationships

import seaborn as sns
import matplotlib.pyplot as plt
#Approval vs. days_to_due
sns.boxplot(x='invoice_approved', y='days_to_due', data=df)
plt.title("Days to Due vs. Approval Status")
plt.xlabel("Invoice Approved")
plt.ylabel("Days to Due")
plt.show()

#Approval by description_cluster
sns.barplot(x='description_cluster', y='invoice_approved', data=df)
plt.title("Approval Rate by Description Cluster")
plt.show()



# #dashboard with streamlit
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
#
# base_path = r'C:\Adi\python_projects\consist\data\cleaned_files'
# file_name = 'kpi_analysis_output.csv'
# file_path = os.path.join(base_path, file_name)
#
# # Read the CSV
# df = pd.read_csv(file_path)
#
# # Load your dataframe (replace with actual data source)
# df = pd.read_csv(file_path, parse_dates=['invoice_date', 'due_date'])
#
# # KPIs
# match_rate = df['match_flag'].mean()
# approval_rate = df['invoice_approved'].mean()
# avg_similarity = df['similarity_score'].mean()
# mean_abs_error = df['abs_error'].mean()
# mean_rel_error = df['relative_error_pct'].mean()
# avg_days_to_due = df['days_to_due'].mean()
#
# st.title("Procurement Matching Dashboard")
#
# # Display KPIs
# col1, col2, col3 = st.columns(3)
# col1.metric("Match Rate", f"{match_rate:.1%}")
# col2.metric("Approval Rate", f"{approval_rate:.1%}")
# col3.metric("Avg Description Similarity", f"{avg_similarity:.2f}")
#
# st.metric("Mean Absolute Error", f"{mean_abs_error:.2f}")
# st.metric("Mean Relative Error (%)", f"{mean_rel_error:.2f}")
# st.metric("Avg Days to Due", f"{avg_days_to_due:.1f}")
#
# # Payment Status Breakdown
# payment_counts = df['payment_status'].value_counts()
# st.bar_chart(payment_counts)
#
# # Correspondence Bin Distribution
# corr_bin_counts = df['correspondence_bin'].value_counts()
# st.bar_chart(corr_bin_counts)
#
# # Filters
# vendor_filter = st.selectbox("Select Vendor ID", options=['All'] + list(df['vendor_id'].unique()))
# if vendor_filter != 'All':
#     filtered_df = df[df['vendor_id'] == vendor_filter]
# else:
#     filtered_df = df
#
# # Display filtered table
# st.dataframe(filtered_df[['invoice_number', 'po_number', 'total_amount', 'payment_status', 'match_flag', 'similarity_score', 'invoice_approved']])
#
# # Recommendations
# st.header("Recommendations")
# st.markdown("""
# - **Improve matching** by tuning similarity thresholds and error tolerances.
# - **Automate approval** for invoices with similarity > 0.85 and relative error < 5%.
# - **Monitor KPIs** regularly to catch process degradation.
# - **Flag invoices** with low similarity or high error for manual review.
# """)

# now use xgboost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Numeric, categorical, and text features
numeric_features = ['quantity', 'unit_price', 'total_amount', 'days_to_due', 'correspondence_index']
categorical_features = ['currency']
text_feature = 'vendor_material_description'

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('txt', TfidfVectorizer(max_features=100), text_feature)
])

# Pipeline using XGBoost classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Hyperparameter grid for tuning
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [3, 6, 10],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__subsample': [0.7, 1.0],
    'clf__colsample_bytree': [0.7, 1.0]
}

# Grid Search with CV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit grid search
grid_search.fit(X, y)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best CV F1 Score:", grid_search.best_score_)

# Evaluate best model on test split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))
