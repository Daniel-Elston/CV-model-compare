import os
import json
import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine, Table, MetaData



def fetch_table_to_parquet(
    db_url,
    table_name,
    data_dir,
    parquet_file_name,
    save_as_excel=False,
    excel_file_name='output.xlsx',
    compression_method='snappy',
    return_metadata=True
    ):
    
    """
    Fetches data from a given table in a database and saves it as a Parquet file.
    
    Parameters:
    - db_url (str): The database connection URL.
    - table_name (str): The name of the table to fetch data from.
    - data_dir (str): The directory where the Parquet file should be saved.
    - parquet_file_name (str, optional): The name of the Parquet file. Defaults to 'intel_image_paths_labels.parq'.
    - compression_method (str, optional): The compression method for the Parquet file. Defaults to 'snappy'.
    - return_metadata (bool, optional): Whether to return metadata about the Parquet file. Defaults to True.
    
    Returns:
    pd.DataFrame: The DataFrame containing the data from the table.
    """
    
    engine = create_engine(db_url)
    metadata = MetaData()

    # Reflect the table from the database
    table = Table(
        table_name,
        metadata,
        autoload_with=engine
    )

    # Query the database
    s = table.select()
    df = pd.read_sql(s, engine)

    # Save the DataFrame as a Parquet file
    file_path = os.path.join(data_dir, f'raw/{parquet_file_name}')
    df.to_parquet(file_path, engine='pyarrow', compression=compression_method)
    pq_file = pq.ParquetFile(file_path)

    # Save the DataFrame as an Excel file if specified
    if save_as_excel:
        excel_file_path = os.path.join(data_dir, f'raw/{excel_file_name}')
        df.to_excel(excel_file_path, index=False)
    
    print(f'Files saved to: data/raw/[file_name]')
    
    
    if return_metadata:
        print(df.info(memory_usage='deep'), '\n')
        print(f'Parquet file metadata: {pq_file.metadata}')    
        print(f'Parquet file saved to: data/raw/{parquet_file_name}')
        print(f'Parquet file size: {os.path.getsize(file_path) / 1e6} MB')
    
    return df


def load_files(evals_files_dict):
    """
    Load evaluation files given a dictionary of model names and paths.
    """
    results = {}
    for model_name, file_path in evals_files_dict.items():
        with open(file_path, 'r') as f:
            results[model_name] = json.load(f)
    return results

def extract_metric_from_results(results, metric_name):
    extracted_metrics = {}
    
    for model_name, i in results.items():

        if isinstance(i, list):
            extracted_metrics[model_name] = [metrics.get(metric_name, None) for metrics in i]
        else:
            extracted_metrics[model_name] = i.get(metric_name, {})
    
    return pd.DataFrame(extracted_metrics)