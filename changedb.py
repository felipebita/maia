import sqlite3
import os

# Step 1: Connect to the database (or create one if it doesn't exist)
app_root = os.path.dirname(os.path.dirname(__file__))  # Go up one level from src folder
db_path = os.path.join(app_root, "databases", "fs_challenge.db")
conn = sqlite3.connect(db_path)

# Step 2: Create a cursor to interact with the database
cursor = conn.cursor()

# Step 3: Specify the table to remove
table_name = "eneva_ts"  # Replace with the name of the table to delete

# Step 4: Drop the table if it exists
try:
    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    print(f"Table '{table_name}' has been removed successfully.")
except sqlite3.Error as e:
    print(f"An error occurred: {e}")

# Step 5: Commit the changes and close the connection
conn.commit()  # Save the changes to the database
conn.close()   # Close the database connection
