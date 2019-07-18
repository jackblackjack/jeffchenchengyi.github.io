import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Merge datasets from messages_filepath and categories_filepath
       into a single dataset
       
    Args:
    -----
    messages_filepath: str, the filepath to the dataset of disaster response 
                       tweets
    categories_filepath: str, the filepath to the dataset of labels for the disaster
                         response tweets
    
    Returns:
    --------
    A dataframe of the merged datasets by id
    """
    return pd.merge(left=pd.read_csv(messages_filepath),
                    right=pd.read_csv(categories_filepath),
                    how='inner',
                    on=['id'])


def clean_data(df):
    """Cleans the merged dataset by splitting the categories feature into separate 
       columns and removing duplicated observations
       
    Args:
    -----
    df: Pandas Dataframe containing the merged messages and categories dataset
    
    Returns:
    --------
    The cleaned dataset after performing all the cleaning steps noted above
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True).apply(lambda row: pd.Series([1 if int(value.split('-')[1]) > 0 else 0 for value in row]), axis=0)
    
    # select the first row of the categories dataframe and
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    # and rename the columns of `categories`
    categories.columns = [value.split('-')[0] for value in df['categories'].iloc[0].split(';')]
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop(df[df.duplicated()].index, axis=0, inplace=True)
    
    # Flag training examples that have no labels
    zero_label_samples = 0
    for row_idx in range(df.shape[0]):
        if np.sum(df.iloc[row_idx, 4:]) == 0:
            zero_label_samples += 1
    print('There are {} samples with no labels...'.format(zero_label_samples))
    
    # Flag any categories that have 0 training examples
    for col in df.iloc[:, 4:].columns:
        if np.sum(df[col]) == 0:
            print('{} has no training samples...'.format(col))
    
    return df


def save_data(df, database_filename):
    """Save the clean dataset into an sqlite database
    
    Args:
    -----
    df: Pandas Dataframe to be saved
    database_filename: str, database file name to save 
    
    Returns:
    --------
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename.split('/')[-1].split('.')[0], engine, index=False)


def main():
    """
    `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()