import pandas as pd
import numpy as np
import json 

def validation(): 
    """Here we're validating enough data for eventual ML training"""
    print('loading data set...')
    try:
        df = pd.read_csv(r'company_emails.csv')
        print('loaded successfully!')
    except FileNotFoundError: 
        try: 
            with open('company_emails.json', 'r') as f: 
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f'loaded json file with {len(df)}')
        except FileNotFoundError: 
            print('no data set... check json or csv files on')
            return None
    rq_col = ['sender', 'subject', 'body', 'label', 'timestamp']
    no_col = [col for col in rq_col if col not in df.columns]

    if no_col: 
        print(f'missing required columns: {no_col}')
        print(f'available columns are: {list(df.columns)}')
        return None
    if len(df) == 0: 
        print('dataset empty!')
        return None
    print("everything loaded correctly")
    print(f'dataset shape: {df.shape}')
    print(f'columns: {list(df.columns)}')
    return df



if __name__ == "__main__": 
    validation()