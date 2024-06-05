import pandas as pd

def split_and_save_batches(input_csv, output_prefix, batch_size):
    """
    Split the input CSV file into batches of specified size and save each batch as a separate CSV file.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_prefix (str): Prefix for the output CSV file names.
    - batch_size (int): Number of rows per batch.

    Returns:
    None
    """
    # Load the CSV
    df = pd.read_csv(input_csv)

    # Take the top 240k rows
    df_top = df.head(240000)

    # Split into batches and save each batch
    for i, batch_start in enumerate(range(0, len(df_top), batch_size)):
        batch = df_top.iloc[batch_start:batch_start + batch_size]
        output_csv = f"{output_prefix}_batch_{i + 1}.csv"
        batch.to_csv(output_csv, index=False)
        print(f"Batch {i + 1} saved to {output_csv}")

# Replace 'your_input_file.csv' with the path to your input CSV file
input_csv_file = 'websites.csv'

# Replace 'output_prefix' with the desired prefix for the output CSV files
output_csv_prefix = 'normal'

# Set the batch size (4k in this case)
batch_size = 4000

split_and_save_batches(input_csv_file, output_csv_prefix, batch_size)
