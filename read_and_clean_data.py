import os
import pandas as pd

# Set your directory containing .txt files
folder_path = r'C:\Adi\python_projects\consist\data'  # Use raw string for Windows paths
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
