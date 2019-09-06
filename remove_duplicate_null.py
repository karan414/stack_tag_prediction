import pandas as pd
import os
import sqlite3
from sqlalchemy import create_engine

input_filename = "./csv/Train.csv"

# Load data into Database from csv
if not os.path.isfile('./db/dataset.db'):
    disk_engine = create_engine('sqlite:///db/dataset.db')
    chunksize = 180000
    j = 0
    index_start = 1
    for df in pd.read_csv(input_filename, names=['Id', 'Title', 'Body', 'Tags'], chunksize=chunksize, iterator=True, encoding='utf-8'):
        df.index += index_start
        j += 1
        print('{} Rows '.format(j*chunksize))
        df.to_sql('stack_data', disk_engine, if_exists='append')
        index_start = df.index[-1] + 1

# Count the number of rows
if os.path.isfile('./db/dataset.db'):
    connection = sqlite3.connect('./db/dataset.db')
    no_of_rows = pd.read_sql_query("""SELECT count(*) FROM stack_data""", connection)
    connection.close()
    print("Number of rows: ", no_of_rows.head())
else:
    print("Fail to open database file")

# Check for duplicates
if os.path.isfile('./db/dataset.db'):
    connection = sqlite3.connect('./db/dataset.db')
    number_of_duplicates = pd.read_sql_query('SELECT Title, Body, Tags, COUNT(*) as Dup_Count FROM stack_data GROUP BY Title, Body, Tags', connection)
    connection.close()
    print(number_of_duplicates.head())
    print("Number of duplicates question: ", no_of_rows['count(*)'].values[0] - number_of_duplicates.shape[0])
    print("Duplicate Percentage: ", (1 - ((number_of_duplicates.shape[0]) / (no_of_rows['count(*)'].values[0]))) * 100, '%')
    print(number_of_duplicates.Dup_Count.value_counts())
else:
    print("Please download the train.db file from drive or run the first to genarate train.db file")

# Counting the tags and adding them into database
number_of_duplicates['Tag_count'] = number_of_duplicates["Tags"].apply(lambda text: len(str(text).split(" ")))
print(number_of_duplicates.head())
print(number_of_duplicates.Tag_count.value_counts())

# Creating new database without duplicates
if not os.path.isfile('./db/no_duplicates.db'):
    disk_no_duplicates = create_engine("sqlite:///db/no_duplicates.db")
    no_duplicates = pd.DataFrame(number_of_duplicates, columns=['Title', 'Body', 'Tags'])
    no_duplicates.to_sql('no_duplicates_data', disk_no_duplicates)

# Dropping the data and storing in the no duplicate database
if os.path.isfile('./db/no_duplicates.db'):
    connection = sqlite3.connect('./db/no_duplicates.db')
    data = pd.read_sql_query("""SELECT * FROM no_duplicates_data""", connection)
    connection.close()
    print(data.head())

    data.drop([data.index[0], data.index[1], data.index[2], data.index[3]], inplace=True)
    print(data.head())

# Count the number of rows
if os.path.isfile('./db/no_duplicates.db'):
    connection = sqlite3.connect('./db/no_duplicates.db')
    no_duplicate_data = pd.read_sql_query("""SELECT * FROM no_duplicates_data""", connection)
    connection.close()
    export_csv = no_duplicate_data.to_csv('./csv/no_duplicates.csv', index=False)
else:
    print("Fail to open database file")
