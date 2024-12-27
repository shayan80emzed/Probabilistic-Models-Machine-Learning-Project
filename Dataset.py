import pandas as pd
import json

# Load the CSV file into a pandas DataFrame
main_dir = "/h/emzed/data/"
csv_file = main_dir + "discharge.csv"
discharge_df = pd.read_csv(csv_file)

# Load the JSONL file into a list of dictionaries
jsonl_file = main_dir + "qa_note.jsonl"
with open(jsonl_file, 'r') as file:
    jsonl_data = [json.loads(line) for line in file]

# Convert JSONL data into a pandas DataFrame
questions_df = pd.DataFrame(jsonl_data)

# Perform an inner join on 'hadm_id'
merged_df = pd.merge(questions_df, discharge_df, on='hadm_id', how='inner')

# Select the desired columns: hadm_id, q, and text
final_df = merged_df[['hadm_id', 'q', 'a', 'text']]

# Save the result to a new CSV file
output_file = main_dir + "qa_discharge.csv"
final_df.to_csv(output_file, index=False)

print(f"Joined CSV file has been saved to {output_file}")
