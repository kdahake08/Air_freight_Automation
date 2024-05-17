import logging
import pandas as pd
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy import text
import psycopg2

# Configure logging
log_file_path = r'C:\Users\Karti\Downloads\TGL_Airfreight_Automation\dist\Output_Files\app.log'
logging.basicConfig(filename=log_file_path, filemode ='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_sheet_path = r'C:\Users\Karti\Downloads\TGL_Airfreight_Automation\dist\input_config_sheet.csv'

try:
    input_sheet_df = pd.read_csv(input_sheet_path)
except FileNotFoundError:
    logging.error(
        "Error: Input sheet not found. Please check the path of the input sheet or ensure that it is not opened elsewhere.")
    raise  # or sys.exit() to exit the script

dbname = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'db_name', 'value'].values[0]
username = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'db_username', 'value'].values[0]
password = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'db_password', 'value'].values[0]
host = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'db_host', 'value'].values[0]
port = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'db_port', 'value'].values[0]

def truncate_table(table_name):

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname=dbname,
        user=username,
        password=password,
        host=host,
        port=port
    )

    # Create a cursor object
    cur = conn.cursor()

    # Truncate the table using a SQL query
    truncate_query = f"TRUNCATE TABLE {table_name} RESTART IDENTITY;"
    cur.execute(truncate_query)

    # Commit the transaction
    conn.commit()

    # Close the cursor and connection
    cur.close()
    conn.close()


# Create the database engine
engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}')

def fetch_raw_data(table_name):
    try:
        df = pd.read_sql_table(table_name, engine)
        return df
    except ValueError as e:
        logging.error(f'Error while retrieving {table_name} from the database, Table not found')


# def compare_data(df, historical_df):
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df['date'] = df['date'].dt.date
#     df.reset_index(drop=True, inplace=True)
#     historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
#     historical_df['date'] = historical_df['date'].dt.date
#     merged_data = pd.merge(historical_df, df, on=['from', 'to', 'airline', 'service'], suffixes=('_old', '_new'),
#                            how='inner')
#     merged_data.reset_index(drop=True, inplace=True)
#     #merged_data.to_csv(r'C:\Users\Karti\Downloads\TGL_Airfreight_Automation\merged.csv', index=True)
#     changed_records = []
#
#     # Iterate through each row in merged_data to check for changes
#     for index, row in merged_data.iterrows():
#         old_date = row['date_old']
#         new_date = row['date_new']
#
#         # Skip rows where both old and new dates are null
#         if pd.isna(old_date) and pd.isna(new_date):
#             continue
#
#         # Compare 'Date' columns for changes
#         if old_date != new_date:
#             #print(index, old_date, new_date)
#             # Append the new row from new_data to changed_records
#             changed_records.append(merged_data.iloc[index].to_dict())
#             #print(changed_records)
#
#     # Convert the list of dictionaries to a DataFrame
#     changed_records_df = pd.DataFrame(changed_records)
#     changed_records_df.drop([])
#     return changed_records_df

def compare_data(df, historical_df):
    # Drop the 'created_at' column from both dataframes
    df.drop('created_at', axis=1, inplace=True)
    historical_df.drop('created_at', axis=1, inplace=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.date
    df.reset_index(drop=True, inplace=True)
    historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
    historical_df['date'] = historical_df['date'].dt.date

    merged_data = pd.merge(historical_df, df, on=['from', 'to', 'airline', 'service'], suffixes=('_old', '_new'),
                           how='inner')
    merged_data.reset_index(drop=True, inplace=True)
    changed_records = []
    columns_list = ['min', 'normal', '+45', '100', '+250', '+300', '+500', '+1000', '+2000', 'date']
    # Iterate through each row in merged_data to check for changes
    for index, row in merged_data.iterrows():
        # Skip rows where both old and new dates are null
        if pd.isna(row['date_old']) and pd.isna(row['date_new']):
            continue

        # Check if any column values are different between old and new data
        if any(row[column + '_old'] != row[column + '_new'] for column in columns_list):
            #logging.info(row)
            changed_records.append(merged_data.iloc[index].to_dict())

    # Convert the list of dictionaries to a DataFrame
    changed_records_df = pd.DataFrame(changed_records)
    return changed_records_df


def parse_log_line(line):
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - (\w+) - (.+)'
    match = re.match(pattern, line)
    if match:
        timestamp_str, level, message = match.groups()
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        return timestamp, level, message
    return None, None, None

def update_log_table(engine, log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    Session = sessionmaker(bind=engine)
    session = Session()

    for line in log_content.strip().split('\n'):
        timestamp, level, message = parse_log_line(line)
        if timestamp and level and message:
            log_entry = {
                'timestamp': timestamp,
                'level': level,
                'message': message
            }
            session.execute(text("""
                INSERT INTO log_entries (timestamp, level, message)
                VALUES (:timestamp, :level, :message)
            """), log_entry)

    session.commit()

