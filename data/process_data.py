#importing libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    
    
    """
    #load the data into two variables
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merging the two datasets
    df = messages.merge(categories, on = 'id')

    return df


def lastString(value):
    return value[-1:]



def clean_data(df):
    """
    
    
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # stroing column names in a list
    category_colnames = [x for x in row]

    # renaming the columns of `categories`
    categories.columns = category_colnames

    #loop through each column
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(np.str)
        categories[column] = categories[column].map(lastString)
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int64)

    # drop the original (uncleaned) categories column from `df`
    del df['categories']

    # concatenate the original dataframe with the new `categories` dataframe
    df2 = pd.concat([df, categories], axis=1)

    #removing duplicates
    df2 = df2.drop_duplicates()

    return df2


def save_data(df, database_filename):
    """
    
    
    """
    engine = create_engine(database_filename)
    df.to_sql('Responses', engine, index=False)


def main():
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